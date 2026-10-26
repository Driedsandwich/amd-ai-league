"""Review regressions; stdlib mocks exercise the GPU interface without torch."""
import contextlib
import http.client
import json
import os
import stat
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from test_box import Sandbox, QUESTION, PRICE, EMPTY, ID_FLAGS, QUERY_FLAGS
import app as entry
from m4agent import backend, retrieval
from m4agent.cli import parse_args
from m4agent.common import atomic_json


class RetrievalRegressions(Sandbox):
    def test_crawl_continues_after_space_in_href(self):
        (self.site / "index.html").write_text(
            '<a href="bad link.html">Products prices</a>'
            '<a href="products.html">Products prices</a>')
        self.start_server()
        origin = self.start_http()
        proc, _ = self.run_app("--query", QUESTION, "--url", origin + "/index.html")
        self.assertEqual(self.saved()["answer"], PRICE)
        self.assertEqual(self.saved()["citations"], [origin + "/prices.html"])
        self.assertEqual(json.loads(proc.stdout), self.saved())

    def test_fetch_contains_request_read_and_parse_exceptions(self):
        url = "http://127.0.0.1/page"
        response = mock.MagicMock()
        response.__enter__.return_value = response
        for error in (http.client.InvalidURL("bad URL"), http.client.IncompleteRead(b"partial"),
                      RuntimeError("unexpected fetch failure")):
            with self.subTest(error=type(error).__name__):
                response.read1.side_effect = error
                with mock.patch("urllib.request.urlopen", return_value=response):
                    self.assertIsNone(retrieval.fetch(url, time.monotonic() + 4))
        with mock.patch.object(retrieval, "http_url", side_effect=ValueError("bad syntax")):
            self.assertIsNone(retrieval.fetch(url, time.monotonic() + 4))


class OutputRegressions(Sandbox):
    def test_missing_option_values_still_publish_and_exit_zero(self):
        for flag in (*QUERY_FLAGS, "--url", "--corpus", "--input", "--index", "--output-dir"):
            with self.subTest(flag=flag):
                proc, _ = self.run_app("--id", "missing-value", flag)
                self.assertEqual(self.saved("missing-value"), EMPTY)
                self.assertEqual(json.loads(proc.stdout), EMPTY)
                (self.output / "missing-value_output.json").unlink()

    def test_error_recovers_id_after_bad_flag_and_all_id_aliases(self):
        for flag in ID_FLAGS:
            for id_args in ([flag, "recovered"], [flag + "=recovered"]):
                with self.subTest(id_args=id_args):
                    proc, _ = self.run_app("--query", *id_args)
                    self.assertEqual(self.saved("recovered"), EMPTY)
                    self.assertEqual(json.loads(proc.stdout), EMPTY)
                    (self.output / "recovered_output.json").unlink()

    def test_missing_input_writes_named_or_default_refusal(self):
        for id_args, query_id in (([], "query"), (["--qid", "missing-batch"], "missing-batch")):
            with self.subTest(query_id=query_id):
                proc, _ = self.run_app("--input", str(self.root / "absent.json"), *id_args)
                self.assertEqual(self.saved(query_id), EMPTY)
                self.assertEqual(json.loads(proc.stdout), EMPTY)

    def test_unexpected_top_level_error_recovers_json_id(self):
        argv = [json.dumps({"id": "json-recovered", "question": QUESTION}),
                "--output-dir=" + str(self.output)]
        with mock.patch.object(entry, "jobs_from_args", side_effect=RuntimeError("unexpected")), \
             mock.patch("builtins.print") as printing:
            self.assertEqual(entry.main(argv), 0)
        self.assertEqual(self.saved("json-recovered"), EMPTY)
        self.assertEqual(json.loads(printing.call_args.args[0]), EMPTY)

    def test_atomic_output_is_readable_before_replace_even_with_private_umask(self):
        path = self.output / "mode.json"
        original_replace = os.replace
        modes = []
        def replacing(source, target):
            modes.append(stat.S_IMODE(Path(source).stat().st_mode))
            original_replace(source, target)
        old_umask = os.umask(0o077)
        try:
            with mock.patch("os.replace", side_effect=replacing):
                atomic_json(path, EMPTY)
                atomic_json(path, EMPTY)
        finally:
            os.umask(old_umask)
        self.assertEqual(modes, [0o644, 0o644])
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
        self.assertEqual(json.loads(path.read_text()), EMPTY)


class ClockAndArgumentRegressions(unittest.TestCase):
    def test_proc_clock_is_trusted_only_within_five_seconds(self):
        for delta in (-100, -0.01, 0, 0.01, 5, 5.01, 100000):
            with self.subTest(delta=delta):
                started = 200000.0
                proc_start = started - delta
                fields = ["0"] * 20
                fields[19] = str(round(proc_start * 100))
                raw = "123 (name with ) parentheses) " + " ".join(fields)
                with mock.patch.object(entry, "STARTED", started), \
                     mock.patch.object(Path, "read_text", return_value=raw), \
                     mock.patch.object(os, "sysconf", return_value=100):
                    self.assertEqual(entry.process_started(),
                                     proc_start if 0 <= delta <= 5 else started)

    def test_bad_proc_stat_uses_module_clock(self):
        with mock.patch.object(Path, "read_text", side_effect=OSError("no proc")):
            self.assertEqual(entry.process_started(), entry.STARTED)

    def test_unknown_option_value_is_not_a_positional_question(self):
        for argv in (["--future", "noise", QUESTION],
                     ["--future=noise", QUESTION],
                     ["--future", "noise", "--id", "x", QUESTION],
                     ["What", "--future", "noise", "is the price?"],
                     ["--future", "--other", "noise", QUESTION]):
            with self.subTest(argv=argv):
                expected = "What is the price?" if argv[0] == "What" else QUESTION
                self.assertEqual(parse_args(argv).query, expected)

    def test_question_flag_ignores_all_positionals_even_when_empty(self):
        for flag in QUERY_FLAGS:
            for value in (QUESTION, ""):
                with self.subTest(flag=flag, value=value):
                    args = parse_args(["--future", "noise", "positional question", flag, value])
                    self.assertEqual(args.query, value)
                    self.assertEqual(args.positional, [])


class BackendRegressions(unittest.TestCase):
    def test_dtype_first_and_legacy_fallback_only_for_type_error(self):
        for error in (None, TypeError("unsupported dtype"), RuntimeError("load failed")):
            with self.subTest(error=error):
                torch = types.SimpleNamespace(cuda=mock.Mock(), bfloat16=object())
                torch.cuda.is_available.return_value = True
                model = mock.Mock()
                factory = mock.Mock()
                factory.from_pretrained.side_effect = [error, model] if error else [model]
                transformers = types.SimpleNamespace(AutoTokenizer=mock.Mock(), AutoModelForCausalLM=factory)
                instance = backend.TransformersBackend()
                with mock.patch.dict(sys.modules, torch=torch, transformers=transformers), \
                     mock.patch.dict(os.environ):
                    if isinstance(error, RuntimeError):
                        with self.assertRaises(RuntimeError):
                            instance.load()
                    else:
                        instance.load()
                        model.to.assert_called_once_with("cuda")
                        model.to.return_value.eval.assert_called_once_with()
                calls = factory.from_pretrained.call_args_list
                self.assertIs(calls[0].kwargs["dtype"], torch.bfloat16)
                self.assertNotIn("torch_dtype", calls[0].kwargs)
                self.assertEqual(len(calls), 2 if isinstance(error, TypeError) else 1)
                if len(calls) == 2:
                    self.assertIs(calls[1].kwargs["torch_dtype"], torch.bfloat16)
                    self.assertNotIn("dtype", calls[1].kwargs)

    def test_generation_explicitly_disables_sampling_config(self):
        instance = backend.TransformersBackend()
        instance.torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        instance.tokenizer = mock.Mock()
        inputs = {"input_ids": types.SimpleNamespace(shape=(1, 12))}
        instance.tokenizer.return_value.to.return_value = inputs
        instance.model = mock.MagicMock()
        instance.generate("question", max_new=8, max_time=10)
        kwargs = instance.model.generate.call_args.kwargs
        self.assertIs(kwargs["do_sample"], False)
        for name in ("temperature", "top_p", "top_k"):
            self.assertIn(name, kwargs)
            self.assertIsNone(kwargs[name])
        self.assertEqual(kwargs["max_new_tokens"], 8)

    def test_short_then_long_warmup(self):
        instance = backend.TransformersBackend()
        with mock.patch.object(instance, "generate") as generate:
            instance.warmup()
        self.assertEqual(generate.call_count, 2)
        short, long = generate.call_args_list
        self.assertLess(len(short.args[0]), 200)
        self.assertEqual(short.kwargs["max_new"], 4)
        self.assertTrue(5500 <= len(long.args[0]) <= 6500)
        self.assertEqual(long.kwargs["max_new"], 8)
        self.assertGreater(long.kwargs["max_time"], 0)
