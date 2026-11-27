"""Review regressions; stdlib mocks exercise the GPU interface without torch."""
import contextlib
import http.client
import io
import json
import os
import stat
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from test_box import APP, Sandbox, QUESTION, PRICE, EMPTY, ID_FLAGS, QUERY_FLAGS
import app as entry
from m4agent import backend, client, diagnostic, retrieval
from m4agent.cli import jobs_from_args, parse_args
from m4agent.common import atomic_json


def output_paths(*ids, directory="output"):
    return [f"{directory}/{item_id}{suffix}" for item_id in ids for suffix in ("_output.json", ".json")]


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
    def assert_single_output(self, proc, item_id):
        paths = output_paths(item_id)
        files, rows = self.assert_outputs(proc, paths)
        self.assertEqual(len(rows), 1)
        self.assertEqual(files, dict.fromkeys(paths, rows[0]))
        self.assertEqual(rows[0]["answer"], PRICE)

    def test_query_id_flag(self):
        self.start_server()
        proc, _ = self.run_app("--query-id", "find_01", "--query", QUESTION, "--corpus", str(self.site))
        self.assert_single_output(proc, "find_01")

    def test_input_file_without_id_uses_stem(self):
        self.start_server()
        path = self.root / "find_01.json"
        path.write_text(json.dumps({"instruction": QUESTION, "link": (self.site / "prices.html").as_uri()}))
        proc, _ = self.run_app("--input", str(path))
        self.assert_single_output(proc, "find_01")

    def test_find_id_and_unknown_question_and_url_flags(self):
        self.start_server()
        proc, _ = self.run_app("--find-id", "find_01", "--summary", "Some short text",
                               "--ask=" + QUESTION, "--target", (self.site / "prices.html").as_uri())
        self.assert_single_output(proc, "find_01")

    def test_positional_input_file_uses_stem(self):
        self.start_server()
        path = self.root / "find_01.json"
        path.write_text(json.dumps({"text": QUESTION, "site": str(self.site / "prices.html")}))
        proc, _ = self.run_app(str(path))
        self.assert_single_output(proc, "find_01")

    def test_directory_inputs_each_use_their_file_stem(self):
        self.start_server()
        inputs = self.root / "inputs"
        inputs.mkdir()
        (inputs / "find_02.json").write_text(json.dumps({"request": QUESTION, "id": "other"}))
        (inputs / "find_01.json").write_text(json.dumps({"find": "", "name": "other"}))
        (inputs / "ignored.txt").write_text(json.dumps({"question": QUESTION}))
        proc, _ = self.run_app("--input-json", str(inputs), "--corpus", str(self.site))
        paths = output_paths("find_01", "find_02")
        files, rows = self.assert_outputs(proc, paths)
        self.assertEqual([row["answer"] for row in rows], ["", PRICE])
        for item_id, row in zip(("find_01", "find_02"), rows):
            self.assertEqual([files[path] for path in output_paths(item_id)], [row, row])

    def test_output_path_adds_exact_file(self):
        self.start_server()
        target = self.root / "some" / "dir" / "x.json"
        proc, _ = self.run_app("--query", QUESTION, "--corpus", str(self.site), "--output", str(target))
        paths = [*output_paths("query"), "some/dir/x.json"]
        files, rows = self.assert_outputs(proc, paths, [self.output, target.parent])
        self.assertEqual(len(rows), 1)
        self.assertEqual(files, dict.fromkeys(paths, rows[0]))
        self.assertEqual(rows[0]["answer"], PRICE)

    def test_find_id_environment_variable(self):
        self.start_server()
        proc, _ = self.run_app("--query", QUESTION, "--corpus", str(self.site),
                               env=dict(self.env, FIND_ID="find_01"))
        self.assert_single_output(proc, "find_01")

    def test_all_id_candidates_fan_out_to_all_directory_aliases(self):
        self.start_server()
        path = self.root / "ignored_stem.json"
        path.write_text(json.dumps({"id": "json-a", "query_id": "json-b", "find_id": "json-c",
                                    "task_id": "json-d", "qid": "json-e", "item_id": "json-f",
                                    "request_id": "json-g", "name": "json-h", "prompt": QUESTION}))
        extra = [self.root / name for name in ("out-dir", "output_dir", "outdir", "results-dir", "result")]
        output_args = [arg for directory in extra for arg in ("--" + directory.name, str(directory))]
        env = dict(self.env, QUERY_ID="env-a", FIND_ID="env-b", TASK_ID="env-c", ITEM_ID="env-d")
        proc, _ = self.run_app("--id", "flag-id", "--task-id", "repeated", "--request-id=repeated",
                               "--envelope", str(path), "word_123", "--opaque=option_7",
                               "--corpus", str(self.site), *output_args, env=env)
        ids = ["flag-id", "repeated", "word_123", "option_7",
               *(f"json-{letter}" for letter in "abcdefgh"), *(f"env-{letter}" for letter in "abcd")]
        directories = [self.output, *extra]
        paths = [path for directory in directories for path in output_paths(*ids, directory=directory.name)]
        files, rows = self.assert_outputs(proc, paths, directories)
        self.assertEqual(len(rows), 1)
        self.assertEqual(files, dict.fromkeys(paths, rows[0]))
        self.assertEqual(rows[0]["answer"], PRICE)

    def test_error_writes_empty_to_all_candidates(self):
        extra = self.root / "extra"
        proc, _ = self.run_app("--query", "--find-id", "find_01", "--item-id", "find_02",
                               "--results-dir", str(extra))
        paths = output_paths("find_01", "find_02") + output_paths("find_01", "find_02", directory="extra")
        files, rows = self.assert_outputs(proc, paths, [self.output, extra])
        self.assertEqual(rows, [EMPTY])
        self.assertTrue(all(payload == EMPTY for payload in files.values()))

    def test_entry_point_aliases_match_app(self):
        self.start_server()
        args = ("--query-id", "find_01", "--query", QUESTION, "--corpus", str(self.site))
        baseline, _ = self.run_app(*args)
        self.assert_single_output(baseline, "find_01")
        expected = {path.name: path.read_bytes() for path in self.output.iterdir()}
        for name in ("main", "agent", "run", "solve", "predict", "inference"):
            with self.subTest(name=name):
                for path in self.output.iterdir():
                    path.unlink()
                command = self.command(*args)
                command[1] = str(APP / f"{name}.py")
                proc = subprocess.run(command, env=self.env, capture_output=True, text=True, timeout=26)
                self.assertEqual(proc.returncode, baseline.returncode, proc.stderr)
                self.assertEqual(proc.stdout, baseline.stdout)
                self.assertEqual(proc.stderr, baseline.stderr)
                self.assertEqual({path.name: path.read_bytes() for path in self.output.iterdir()}, expected)

    def test_missing_option_values_still_publish_and_exit_zero(self):
        for flag in (*QUERY_FLAGS, "--url", "--corpus", "--input", "--index", "--output-dir"):
            with self.subTest(flag=flag):
                proc, _ = self.run_app("--id", "missing-value", flag)
                self.assertEqual(self.saved("missing-value"), EMPTY)
                self.assertEqual(json.loads(proc.stdout), EMPTY)
                for path in output_paths("missing-value"):
                    (self.root / path).unlink()

    def test_error_recovers_id_after_bad_flag_and_all_id_aliases(self):
        for flag in ID_FLAGS:
            for id_args in ([flag, "recovered"], [flag + "=recovered"]):
                with self.subTest(id_args=id_args):
                    proc, _ = self.run_app("--query", *id_args)
                    self.assertEqual(self.saved("recovered"), EMPTY)
                    self.assertEqual(json.loads(proc.stdout), EMPTY)
                    for path in output_paths("recovered"):
                        (self.root / path).unlink()

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


class StartupRegressions(Sandbox):
    def test_missing_socket_launches_once_after_placeholders_and_keeps_fallback(self):
        batch = self.root / "batch.json"
        batch.write_text(json.dumps([{"id": "first", "query_id": "alternate", "query": QUESTION},
                                     {"id": "second", "query": QUESTION}]))
        paths = [self.root / path for path in output_paths("first", "alternate", "second")]
        launches = self.root / "launches.log"
        waits = self.root / "waits.log"
        wait_ready = client.wait_ready

        # These logs record the calls made by the forked worker for the parent test.
        def launch(command, **kwargs):
            self.assertEqual(command, [sys.executable, str(APP / "supervise.py")])
            self.assertTrue(kwargs["start_new_session"])
            self.assertFalse(Path(self.sock).exists())
            self.assertEqual([json.loads(path.read_text()) for path in paths], [EMPTY] * len(paths))
            with launches.open("a") as log:
                log.write("launch\n")
            raise OSError("simulated launch failure")

        def record_wait(deadline):
            with waits.open("a") as log:
                log.write(f"{deadline - time.monotonic()}\n")
            return wait_ready(deadline)

        argv = ["--input", str(batch), "--corpus", str(self.site), "--output-dir", str(self.output)]
        stdout = io.StringIO()
        started = time.monotonic()
        with mock.patch.dict(os.environ, self.env, clear=True), contextlib.redirect_stdout(stdout), \
             mock.patch.object(client.subprocess, "Popen", side_effect=launch), \
             mock.patch.object(client, "wait_ready", side_effect=record_wait), \
             mock.patch.object(entry, "process_started", return_value=started):
            self.assertEqual(entry.main(argv), 0)
        self.assertLess(time.monotonic() - started, 24)
        self.assertEqual(launches.read_text().splitlines(), ["launch"])
        wait, = map(float, waits.read_text().splitlines())
        self.assertGreater(wait, 0)
        self.assertLessEqual(wait, 0.25)
        fallback = {"answer": PRICE, "citations": ["prices.html"], "confidence": 0.55}
        self.assertEqual([json.loads(path.read_text()) for path in paths], [fallback] * len(paths))
        self.assertEqual([json.loads(line) for line in stdout.getvalue().splitlines()], [fallback] * 2)


class DiagnosticRegressions(Sandbox):
    def command(self, *args):
        # Keep the production locking and process boundaries, with a private counter and probe.
        wrapper = """
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv.pop(1))
import app
from m4agent import diagnostic
diagnostic.COUNTER_PATH = sys.argv.pop(1)
with patch.object(diagnostic, "network_available", return_value=0):
    raise SystemExit(app.main())
"""
        return [sys.executable, "-c", wrapper, str(APP), str(self.root / "diag.count"),
                "--output-dir", str(self.output), *args]

    def test_diagnostic_first_two_calls_only_and_keeps_values_private(self):
        self.start_server()
        env = dict(self.env, X052_DIAG="1")
        index, _ = self.run_app("--index", str(self.site), env=env)
        self.assertEqual(index.stdout, "")
        self.assertFalse((self.root / "diag.count").exists())

        url = (self.site / "prices.html").as_uri()
        path = self.root / "request.json"
        path.write_text(json.dumps({"question": QUESTION, "url": url, url: "secret contents"}))
        paths = output_paths("find_01", "item_2", "request")
        extra = "my private flag value"
        for call in range(3):
            proc, _ = self.run_app("--input", str(path), "--find-id", "find_01", "item_2",
                                   "--corpus", str(self.site), "--future", extra, env=env)
            files, rows = self.assert_outputs(proc, paths)
            self.assertEqual(len(rows), 1)
            self.assertEqual(files, dict.fromkeys(paths, rows[0]))
            if call < 2:
                self.assertEqual(rows[0], {
                    "answer": "DIAG flags=--corpus,--find-id,--future,--input,--output-dir "
                              "pos=1 inkeys=question,url url=file corpus=1 net=0",
                    "citations": ["diag"], "confidence": 0.0,
                })
                for private in (QUESTION, url, str(path), str(self.site), extra,
                                "secret contents", "find_01", "item_2"):
                    self.assertNotIn(private, rows[0]["answer"])
            else:
                self.assertEqual(rows[0]["answer"], PRICE)
        self.assertEqual((self.root / "diag.count").read_text(), "2")

    def test_diagnostic_off_never_claims_a_slot(self):
        self.start_server()
        for env in (self.env, dict(self.env, X052_DIAG="0"), self.env):
            proc, _ = self.run_app("--query", QUESTION, "--corpus", str(self.site), env=env)
            paths = output_paths("query")
            files, rows = self.assert_outputs(proc, paths)
            self.assertEqual(len(rows), 1)
            self.assertEqual(files, dict.fromkeys(paths, rows[0]))
            self.assertEqual(rows[0]["answer"], PRICE)
        self.assertFalse((self.root / "diag.count").exists())

    def test_probe_uses_fixed_endpoint_and_one_second_timeout(self):
        with mock.patch.object(diagnostic.socket, "socket") as opening:
            conn = opening.return_value.__enter__.return_value
            self.assertEqual(diagnostic.network_available(), 1)
            opening.assert_called_once_with(diagnostic.socket.AF_INET, diagnostic.socket.SOCK_STREAM)
            conn.settimeout.assert_called_once_with(1)
            conn.connect.assert_called_once_with(("1.1.1.1", 443))
            conn.connect.side_effect = TimeoutError
            self.assertEqual(diagnostic.network_available(), 0)


class ClockAndArgumentRegressions(unittest.TestCase):
    def test_default_directory_is_used_without_output_dir_flag(self):
        with mock.patch.dict(os.environ, clear=True):
            job, = jobs_from_args(parse_args(["--out-dir", "/alternate", "--find-id", "find_01"]))
        self.assertEqual(job["outputs"], [Path("/app/output/find_01_output.json"),
                                          Path("/alternate/find_01_output.json"),
                                          Path("/app/output/find_01.json"),
                                          Path("/alternate/find_01.json")])

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
