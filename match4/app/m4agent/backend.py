"""Text-only backends. Heavy dependencies are imported exclusively in load()."""
import os
import time
from pathlib import Path


class StubBackend:
    def load(self):
        time.sleep(float(os.environ.get("X052_STUB_LOAD_SLEEP", "0")))
        if os.environ.get("X052_STUB_FAIL_LOAD") == "1":
            raise RuntimeError("requested stub load failure")

    def warmup(self):
        pass

    def generate(self, prompt, max_new=64, max_time=1):
        # Intentionally ignores max_time: exercises client deadlines on a hung backend.
        time.sleep(float(os.environ.get("X052_STUB_SLEEP", "0")))
        if os.environ.get("X052_STUB_LOG"):
            with open(os.environ["X052_STUB_LOG"], "a", encoding="utf-8") as stream:
                stream.write(prompt + "\n")
        answer_file = os.environ.get("X052_STUB_ANSWER_FILE")
        if answer_file:
            return Path(answer_file).read_text(encoding="utf-8")
        return os.environ.get("X052_STUB_ANSWER", "{}")


class TransformersBackend:
    def load(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm GPU unavailable; CPU fallback is forbidden")
        self.torch = torch
        model_dir = os.environ.get("X052_MODEL_DIR", "/models/Qwen3-4B-Instruct-2507")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch.bfloat16, local_files_only=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, local_files_only=True,
            )
        self.model = model.to("cuda")
        self.model.eval()

    def warmup(self):
        self.generate("Return an empty JSON object.", max_new=4, max_time=20)
        # Exercise passage-sized prefill kernels during startup, before queries arrive.
        self.generate("Read the following context and return an empty JSON object.\n" +
                      ("Context contains reference information. " * 150),
                      max_new=8, max_time=60)

    def generate(self, prompt, max_new=64, max_time=1):
        if max_time <= 0:
            return ""
        end = time.monotonic() + max_time
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=8192).to("cuda")
        remaining = end - time.monotonic()
        if remaining <= 0:
            return ""
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                max_new_tokens=min(64, max(1, int(max_new))),
                max_time=remaining, pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


def make_backend():
    name = os.environ.get("X052_BACKEND", "transformers").lower()
    if name == "stub":
        return StubBackend()
    if name == "transformers":
        return TransformersBackend()
    raise ValueError("unknown backend: " + name)
