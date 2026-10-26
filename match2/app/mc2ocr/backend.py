"""模型の層。StubBackend（試験用）と TransformersBackend（本番）は同じ形。

形: load() / warmup() / read_text(image, prompt, max_new_tokens, max_time) -> str
重い import（torch・transformers）は TransformersBackend.load() の中だけ（遅延 import）。
"""
from __future__ import annotations

import os
import time
from typing import Optional

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # 第一候補。像の中の /models/<名前> に置き、X052_MODEL_DIR で指す


class StubBackend:
    """模型なしで流れを試す。返す文は X052_STUB_ANSWER_FILE（毎回読み直す）→ X052_STUB_ANSWER → 既定の順に決まる。

    試験用の環境変数: X052_STUB_FAIL_LOAD=1（読み込み失敗）・X052_STUB_LOAD_SLEEP=秒（読み込みが遅い）・
    X052_STUB_SLEEP=秒（生成が遅い・max_time で止まり空を返す）・X052_STUB_LOG=ファイル（受けた画像の大きさとモードを足し書き）。
    """

    has_vision = True

    def __init__(self, answer: Optional[str] = None):
        self.answer = answer
        self.warnings: list = []

    def load(self) -> None:
        time.sleep(float(os.environ.get("X052_STUB_LOAD_SLEEP", "0") or 0))
        if os.environ.get("X052_STUB_FAIL_LOAD") == "1":
            raise RuntimeError("stub: model load failed on purpose")

    def warmup(self) -> None:
        return None

    def _answer(self) -> str:
        if self.answer is not None:
            return self.answer
        f = os.environ.get("X052_STUB_ANSWER_FILE")
        if f:
            try:
                with open(f, encoding="utf-8") as fh:
                    return fh.read().rstrip("\n")
            except OSError:
                pass
        return os.environ.get("X052_STUB_ANSWER", "stub-text")

    def read_text(self, image, prompt: str, max_new_tokens: int = 64, max_time: Optional[float] = None) -> str:
        log = os.environ.get("X052_STUB_LOG")
        if log:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"{image.size[0]}x{image.size[1]} {image.mode}\n")
        slow = float(os.environ.get("X052_STUB_SLEEP", "0") or 0)
        if slow:
            time.sleep(min(slow, max_time) if max_time is not None else slow)
            if max_time is not None and slow > max_time:
                return ""  # 時間切れで途中までしか出ない
        return self._answer()


class TransformersBackend:
    """/models の下の視覚つき言語モデルを GPU に明示して置く。手元（Mac）では読み込んでいない（GPU の会で試す）。

    模型の場所は環境変数 X052_MODEL_DIR。無いときだけ /models の下のフォルダが 1 つならそれを使い、warnings に残す。
    GPU が無いときは CPU へ落とさず例外（GPU を使わない解は認められないため）。
    """

    has_vision = True

    def __init__(self, model_dir: Optional[str] = None):
        self.warnings: list = []
        self.model_dir = model_dir or os.environ.get("X052_MODEL_DIR") or self._fallback_model_dir()
        self.model = None
        self.processor = None
        self.device = os.environ.get("X052_DEVICE", "cuda")

    def _fallback_model_dir(self) -> str:
        root = "/models"
        names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
        if not names:
            raise FileNotFoundError("X052_MODEL_DIR is not set and /models has no model directory")
        self.warnings.append(f"X052_MODEL_DIR not set; using /models/{names[0]} (candidates: {names}).")
        return os.path.join(root, names[0])

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch  # 遅延 import
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError:  # transformers が古い等。汎用の読み口へ
            from transformers import AutoModelForImageTextToText as ModelCls
            self.warnings.append("Qwen3VLForConditionalGeneration not importable; using AutoModelForImageTextToText")
        if not torch.cuda.is_available():  # ROCm 版 torch でも cuda と名乗る
            raise RuntimeError("GPU not available; refusing to fall back to CPU")
        self.processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = ModelCls.from_pretrained(
            self.model_dir, torch_dtype=torch.bfloat16, local_files_only=True
        ).to(self.device)  # 明示的に GPU へ
        self.model.eval()

    def warmup(self) -> None:
        """最初の 1 回の遅さ（カーネルの準備）を起動の中に済ませる。失敗しても常駐は続ける。"""
        try:
            from PIL import Image

            img = Image.new("RGB", (256, 128), (255, 255, 255))
            self.read_text(img, "Read the text in the image.", 4, 20.0)
        except Exception as e:  # noqa: BLE001
            self.warnings.append(f"warmup failed: {type(e).__name__}: {e}")

    def read_text(self, image, prompt: str, max_new_tokens: int = 64, max_time: Optional[float] = None) -> str:
        import torch

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.device)
            kw = {"max_time": float(max_time)} if max_time is not None else {}
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **kw)
            new = out[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()
        except Exception:
            try:
                torch.cuda.empty_cache()  # メモリ不足の後も常駐を使い続けるため
            except Exception:  # noqa: BLE001
                pass
            raise


def make_backend(name: Optional[str] = None):
    name = (name or os.environ.get("X052_BACKEND") or "transformers").lower()
    if name == "stub":
        return StubBackend()
    if name == "transformers":
        return TransformersBackend()
    raise ValueError(f"unknown backend: {name}")
