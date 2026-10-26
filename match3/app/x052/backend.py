"""模型の層。StubBackend（試験用）と TransformersBackend（本番）は同じ形。

形: generate(prompt, max_new_tokens, max_time=None) -> str / describe_image(path, max_time=None) -> str / has_vision: bool
重い import（torch・transformers）は TransformersBackend.load() の中だけ（遅延 import）。
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # 第一候補（/models/<名前> に置く。名前は X052_MODEL_DIR で指す）


class StubBackend:
    """模型なしで流れを試す。プロンプトの最初の塊のファイルを引用に挙げ、決まった答えを返す。

    試験用の環境変数: X052_STUB_FAIL_LOAD=1（読み込み失敗）・X052_STUB_LOAD_SLEEP=秒（読み込みが遅い）・X052_STUB_VISION=1（画像を読める顔）・
    X052_STUB_SLEEP=秒（生成が遅い・max_time で止まる）・X052_STUB_HANG=秒（max_time を無視して固まる）。
    """

    def __init__(self, answer: Optional[str] = None):
        self.answer = answer if answer is not None else os.environ.get("X052_STUB_ANSWER", "stub-answer")
        self.has_vision = os.environ.get("X052_STUB_VISION") == "1"
        self.warnings: list = []

    def load(self) -> None:
        time.sleep(float(os.environ.get("X052_STUB_LOAD_SLEEP", "0") or 0))  # 起動が遅い模型の試験用
        if os.environ.get("X052_STUB_FAIL_LOAD") == "1":
            raise RuntimeError("stub: model load failed on purpose")

    def generate(self, prompt: str, max_new_tokens: int = 128, max_time: Optional[float] = None) -> str:
        hang = float(os.environ.get("X052_STUB_HANG", "0") or 0)
        if hang:
            time.sleep(hang)  # max_time を無視する（最後の砦の試験用）
        slow = float(os.environ.get("X052_STUB_SLEEP", "0") or 0)
        if slow:
            time.sleep(min(slow, max_time) if max_time is not None else slow)
            if max_time is not None and slow > max_time:
                return ""  # 時間切れで途中までしか出ない
        paths = re.findall(r"^### FILE: (.+?) \[", prompt, flags=re.M)
        if not paths or not self.answer:
            return json.dumps({"answerable": False, "answer": "", "sources": []})
        return json.dumps({"answerable": True, "answer": self.answer, "sources": [paths[0]]})

    def describe_image(self, path: str, max_time: Optional[float] = None) -> str:
        name = os.path.basename(path)
        if "slow" in name:
            time.sleep(min(float(os.environ.get("X052_STUB_IMAGE_SLEEP", "5")), max_time or 1e9))
            return ""  # 時間切れ
        return f"IMAGE TEXT of {name}"


class TransformersBackend:
    """/models の下の視覚つき言語モデルを GPU に明示して置く。手元（Mac）では試していない。

    模型の場所は環境変数 X052_MODEL_DIR を必ず使う。無いときだけ /models の下のフォルダが 1 つなら
    それを使い、warnings に残す（常駐が INDEX_DIR/warnings.json へ書く）。
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
        self.warnings.append(f"X052_MODEL_DIR not set; using /models/{names[0]} "
                             f"(candidates: {names}). Set X052_MODEL_DIR explicitly.")
        return os.path.join(root, names[0])

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch  # 遅延 import
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls  # 第一候補の模型の型
        except ImportError:  # transformers が古い等。汎用の読み口へ（GPU の会で版を確かめる）
            from transformers import AutoModelForImageTextToText as ModelCls
            self.warnings.append("Qwen3VLForConditionalGeneration not importable; using AutoModelForImageTextToText")
        if not torch.cuda.is_available():  # ROCm 版 torch でも cuda と名乗る
            raise RuntimeError("GPU not available; refusing to fall back to CPU")
        self.processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = ModelCls.from_pretrained(
            self.model_dir, torch_dtype=torch.bfloat16, local_files_only=True
        ).to(self.device)  # 明示的に GPU へ
        self.model.eval()

    def _run(self, messages, max_new_tokens: int, max_time: Optional[float]) -> str:
        import torch

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        kw = {"max_time": float(max_time)} if max_time is not None else {}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **kw)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0].strip()

    def generate(self, prompt: str, max_new_tokens: int = 128, max_time: Optional[float] = None) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._run(messages, max_new_tokens, max_time)

    def describe_image(self, path: str, max_time: Optional[float] = 20.0) -> str:
        from PIL import Image  # transformers の依存として入る

        img = Image.open(path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": (
                "Transcribe all text, labels, pin names, table cells and numbers visible in this "
                "image exactly as printed, one item per line. Keep each label next to its value "
                "(for example 'SIGNAL_NAME = A07'). Then add one line starting with 'DESCRIPTION:' "
                "summarising what the image shows.")},
        ]}]
        return self._run(messages, 400, max_time)


def make_backend(name: Optional[str] = None):
    name = (name or os.environ.get("X052_BACKEND") or "transformers").lower()
    if name == "stub":
        return StubBackend()
    if name == "transformers":
        return TransformersBackend()
    raise ValueError(f"unknown backend: {name}")
