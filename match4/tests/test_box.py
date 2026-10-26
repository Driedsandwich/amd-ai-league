"""CPU-only integration and contract tests. Every runtime artifact stays in mc4/."""
import ast
import functools
import http.server
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))
from m4agent import answer, client, retrieval
from m4agent.cli import CORPUS_FLAGS, ID_FLAGS, QUERY_FLAGS, URL_FLAGS, jobs_from_args, parse_args
from m4agent.common import first_object, normalise, result

PRODUCT = "AsterQuill-R7"
PRICE = "$137.42"
OTHER_PRODUCT = "BirchBolt-K2"
OTHER_PRICE = "$286.13"
LAUNCH = "2087-06-19"
FIXTURE_VALUES = [PRODUCT, PRICE, OTHER_PRODUCT, OTHER_PRICE, LAUNCH]
QUESTION = "What is the price of " + PRODUCT + "?"
EMPTY = {"answer": "", "citations": [], "confidence": 0.0,
         "sources": [], "source_url": ""}


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="run-", dir=ROOT)
        self.root = Path(self.temp.name)
        self.site = self.root / "site"
        self.site.mkdir()
        (self.site / "index.html").write_text(
            '<title>Welcome</title><a href="products.html">Products and prices</a>')
        (self.site / "products.html").write_text(
            '<title>Products</title><a href="prices.html">Price table</a>')
        (self.site / "prices.html").write_text(
            '<title>Product prices</title><script>untrusted script</script><nav>navigation</nav>'
            '<table><tr><th>Product</th><th>Price</th></tr><tr><td>' + PRODUCT + '</td><td>'
            + PRICE + '</td></tr><tr><td>' + OTHER_PRODUCT + '</td><td>' + OTHER_PRICE
            + '</td></tr></table><p>Launch date: ' + LAUNCH + '</p><footer>footer text</footer>')
        self.output = self.root / "output"
        self.sock = str(self.root / "m.sock")
        self.response = self.root / "response.txt"
        self.response.write_text(json.dumps({"answer": PRICE, "source": 1, "confidence": 0.93}))
        self.env = dict(os.environ, X052_BACKEND="stub", X052_SOCKET=self.sock,
                        X052_INDEX_DIR=str(self.root / "index"),
                        X052_STUB_ANSWER_FILE=str(self.response),
                        X052_STUB_LOG=str(self.root / "prompts.log"),
                        PYTHONPATH=str(ROOT / "tests" / "guard"), PYTHONDONTWRITEBYTECODE="1",
                        NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
        for key in ("X052_STUB_SLEEP", "X052_STUB_FAIL_LOAD", "X052_STUB_LOAD_SLEEP", "X052_TEST_NO_INET"):
            self.env.pop(key, None)
        self.processes = []
        self.http = None

    def tearDown(self):
        # Includes detached supervisors launched by app.py --index.
        pid_file = Path(self.sock + ".supervisor.pid")
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
            end = time.monotonic() + 2
            while pid_file.exists() and time.monotonic() < end:
                time.sleep(0.02)
        for proc in self.processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=2)
        if self.http:
            self.http.shutdown()
            self.http.server_close()
            self.http_thread.join(timeout=2)
        self.temp.cleanup()

    def start_server(self, **settings):
        env = dict(self.env, **settings)
        proc = subprocess.Popen([sys.executable, str(APP / "server.py")], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        self.processes.append(proc)
        with mock.patch.dict(os.environ, env):
            self.assertTrue(client.wait_ready(time.monotonic() + 4), "stub server did not become ready")
        return proc

    def start_http(self, handler=QuietHandler):
        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                    functools.partial(handler, directory=str(self.site)))
        self.http_thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.http_thread.start()
        return "http://127.0.0.1:" + str(self.http.server_port)

    def command(self, *args):
        return [sys.executable, str(APP / "app.py"), "--output-dir", str(self.output), *args]

    def run_app(self, *args, env=None, timeout=26):
        started = time.monotonic()
        proc = subprocess.run(self.command(*args), env=env or self.env,
                              capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - started
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc, elapsed

    def saved(self, query_id="query"):
        return json.loads((self.output / (query_id + "_output.json")).read_text())


class IntegrationTests(Sandbox):
    def test_local_multihop_site_stub_and_resident_reuse(self):
        server = self.start_server()
        origin = self.start_http()
        for query_id in ("first", "second"):
            proc, elapsed = self.run_app("--query", QUESTION, "--query-id", query_id,
                                         "--url", origin + "/index.html", "--future-option", "ignored")
            saved = self.saved(query_id)
            self.assertEqual(saved["answer"], PRICE)
            self.assertEqual(saved["citations"], [origin + "/prices.html"])
            self.assertEqual(saved["confidence"], 0.93)  # Proves the stub actually answered.
            self.assertEqual(saved["sources"], saved["citations"])
            self.assertEqual(saved["source_url"], saved["citations"][0])
            self.assertEqual(json.loads(proc.stdout), saved)
            self.assertLess(elapsed, 26)
        with mock.patch.dict(os.environ, self.env):
            self.assertEqual(client.call({"op": "ping"}, 1)["pid"], server.pid)
        prompt = (self.root / "prompts.log").read_text()
        self.assertIn("candidate_values", prompt)
        self.assertNotIn("untrusted script", prompt)

    def test_index_autostarts_server_and_offline_query_without_network(self):
        (self.site / "note.txt").write_text("Plain notes")
        (self.site / "guide.md").write_text("Markdown notes")
        (self.site / "data.json").write_text('{"note":"JSON notes"}')
        (self.site / "other.htm").write_text("<p>HTML notes</p>")
        (self.site / "ignored.bin").write_bytes(b"\x00")
        no_net = dict(self.env, X052_TEST_NO_INET="1")
        proc, _ = self.run_app("--index", str(self.site), env=no_net)
        self.assertIn("indexed 7 files", proc.stderr)
        indexes = list((self.root / "index").glob("*.json"))
        self.assertEqual(len(indexes), 1)
        data = json.loads(indexes[0].read_text())
        self.assertEqual(len(data["pages"]), 7)
        # Demonstrate that the saved index is usable without rescanning source files.
        (self.site / "prices.html").unlink()
        proc, _ = self.run_app("--corpus", str(self.site), "--query", QUESTION, env=no_net)
        self.assertEqual(self.saved()["answer"], PRICE)
        self.assertEqual(self.saved()["citations"], ["prices.html"])
        self.assertEqual(json.loads(proc.stdout), self.saved())

    def test_json_positional_and_file_path_mode(self):
        self.start_server()
        payload = {"question": QUESTION, "id": "json-one", "url": str(self.site / "prices.html")}
        proc, _ = self.run_app(json.dumps(payload), env=dict(self.env, X052_TEST_NO_INET="1"))
        self.assertEqual(self.saved("json-one")["answer"], PRICE)
        self.assertEqual(json.loads(proc.stdout), self.saved("json-one"))

    def test_json_and_jsonl_batches(self):
        self.start_server()
        items = [{"id": "item-a", "query": QUESTION}, {"query_id": "item-b", "question": QUESTION}]
        for suffix in ("json", "jsonl"):
            with self.subTest(suffix=suffix):
                path = self.root / ("batch." + suffix)
                path.write_text(json.dumps(items) if suffix == "json" else "\n".join(map(json.dumps, items)))
                proc, _ = self.run_app("--input", str(path), "--corpus", str(self.site))
                rows = [json.loads(line) for line in proc.stdout.splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows, [self.saved("item-a"), self.saved("item-b")])
                self.assertTrue(all(row["answer"] == PRICE for row in rows))

    def test_hanging_backend_publishes_early_placeholder_and_fallback(self):
        self.start_server(X052_STUB_SLEEP="60")
        started = time.monotonic()
        proc = subprocess.Popen(self.command("--query", QUESTION, "--corpus", str(self.site)),
                                env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        self.processes.append(proc)
        path = self.output / "query_output.json"
        end = time.monotonic() + 2
        while not path.exists() and time.monotonic() < end:
            time.sleep(0.005)
        self.assertTrue(path.exists(), "placeholder missing during inference")
        early = json.loads(path.read_text())
        self.assertIn("answer", early)
        self.assertIn("citations", early)
        stdout, stderr = proc.communicate(timeout=26)
        self.assertEqual(proc.returncode, 0, stderr)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 24)
        self.assertEqual(self.saved()["answer"], PRICE)
        self.assertEqual(self.saved()["confidence"], 0.55)
        self.assertEqual(self.saved()["citations"], ["prices.html"])
        self.assertEqual(json.loads(stdout), self.saved())

    def test_unreachable_loopback_returns_well_formed_refusal(self):
        self.start_server()
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            port = unused.getsockname()[1]
            # Bound but deliberately not listening, so connection fails immediately.
            proc, elapsed = self.run_app("--question", QUESTION, "--url", f"http://127.0.0.1:{port}/")
        self.assertLess(elapsed, 26)
        self.assertEqual(self.saved(), EMPTY)
        self.assertEqual(json.loads(proc.stdout), EMPTY)

    def test_placeholder_survives_external_kill_during_fetch(self):
        self.start_server()
        entered = threading.Event()
        release = threading.Event()

        class Stalled(QuietHandler):
            def do_GET(self):
                entered.set()
                release.wait(5)

        origin = self.start_http(Stalled)
        proc = subprocess.Popen(self.command("--query", QUESTION, "--url", origin), env=self.env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.processes.append(proc)
        try:
            self.assertTrue(entered.wait(3))
            self.assertEqual(self.saved(), EMPTY)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)
            self.assertEqual(self.saved(), EMPTY)
        finally:
            release.set()

    def test_parent_deadline_terminates_a_stalled_worker(self):
        # Exercise the production parent watchdog without depending on DNS or internet.
        spec = importlib.util.spec_from_file_location("m4_entry", APP / "app.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        argv = ["--query", QUESTION, "--corpus", str(self.site), "--output-dir", str(self.output)]
        def hung(jobs, deadline):
            time.sleep(60)
        started = time.monotonic()
        with mock.patch.object(entry, "answer_jobs", hung), \
             mock.patch.object(entry, "process_started", return_value=time.monotonic() - 22), \
             mock.patch("builtins.print"):
            self.assertEqual(entry.main(argv), 0)
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(self.saved(), EMPTY)

    def test_no_network_calls_for_folder_or_missing_folder(self):
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("HTTP forbidden")), \
             mock.patch("socket.create_connection", side_effect=AssertionError("TCP forbidden")), \
             mock.patch("socket.getaddrinfo", side_effect=AssertionError("DNS forbidden")), \
             mock.patch.dict(os.environ, self.env):
            pages = retrieval.gather(QUESTION, str(self.site), None, time.monotonic() + 3)
            self.assertEqual(answer.fallback(QUESTION, pages)[0]["answer"], PRICE)
            self.assertEqual(retrieval.gather(QUESTION, None, str(self.site / "missing"),
                                              time.monotonic() + 3), [])

    def test_bad_file_does_not_abort_index(self):
        original = Path.open
        def opening(path, *args, **kwargs):
            if path.name == "products.html":
                raise PermissionError("simulated unreadable file, including when run as root")
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "open", opening):
            pages = list(retrieval.read_folder(self.site))
        self.assertEqual({p["source"] for p in pages}, {"index.html", "prices.html"})

    def test_bad_model_output_retains_rule_answer(self):
        self.start_server()
        for text in ('not JSON', '{"answer":"imaginary value","source":1}',
                     '{"answer":"' + PRICE + '","source":999}'):
            self.response.write_text(text)
            self.run_app("--query", QUESTION, "--corpus", str(self.site))
            self.assertEqual(self.saved()["answer"], PRICE)
            self.assertEqual(self.saved()["confidence"], 0.55)


class UnitTests(unittest.TestCase):
    def test_all_flag_aliases(self):
        for aliases, dest, value in [(QUERY_FLAGS, "query", QUESTION), (ID_FLAGS, "query_id", "qid-special"),
                                     (URL_FLAGS, "url", "http://127.0.0.1/"), (CORPUS_FLAGS, "corpus", "pages")]:
            for flag in aliases:
                with self.subTest(flag=flag):
                    self.assertEqual(getattr(parse_args([flag, value, "--unknown", "unused"]), dest), value)
        self.assertEqual(parse_args([QUESTION]).query, QUESTION)
        self.assertEqual(parse_args(["--query", QUESTION, "--querying", "noise"]).query, QUESTION)
        parsed = parse_args([json.dumps({"prompt": QUESTION, "task_id": "pos-id", "base_url": "site"})])
        self.assertEqual((parsed.query, parsed.query_id, parsed.url), (QUESTION, "pos-id", "site"))
        self.assertEqual(jobs_from_args(parse_args([QUESTION]))[0]["id"], "query")

    def test_clean_html_tables_title_canonical_and_dropped_elements(self):
        page = retrieval.page_from_text('<title>Useful title</title><link rel="canonical" href="/official">'
            '<style>css trash</style><script>bad instruction</script><nav><p>navigation</p></nav>'
            '<table><tr><td>Widget</td><td>Value</td></tr></table><footer>footer trash</footer>',
            "http://127.0.0.1/start", True)
        self.assertEqual(page["title"], "Useful title")
        self.assertEqual(page["canonical"], "http://127.0.0.1/official")
        self.assertEqual(page["text"], "Widget | Value")

    def test_answer_types_and_candidates(self):
        cases = [("price", "What is the price?", PRICE),
                 ("date", "What is the release date?", LAUNCH),
                 ("year", "What year was it founded?", "2087"),
                 ("version", "What version is current?", "v7.82.13"),
                 ("email", "What is the email?", "custodian@asterquill.test"),
                 ("phone", "What is the phone?", "+81 55 874 2323"),
                 ("url", "What is the website URL?", "https://asterquill.test/docs"),
                 ("percentage", "What percentage is the discount?", "27.35%"),
                 ("count", "How many ports are there?", "37"),
                 ("number", "What is the maximum temperature?", "123"),
                 ("name", "Who is the director?", "Fable Arbor"),
                 ("yes/no", "Is the product available?", "yes"),
                 ("other", "What is the finish?", "Cerulean")]
        for kind, question, value in cases:
            with self.subTest(kind=kind):
                self.assertEqual(answer.guess_type(question), kind)
                p = [{"text": question + " | " + value, "title": "Record", "source": "record.txt", "score": 1}]
                self.assertIn(value, [c["value"] for c in answer.candidates(question, p)])

    def test_grounding_refusal_normalisation_and_two_sources(self):
        passages = [{"source": "one.txt", "title": "", "text": "Name: Fable"},
                    {"source": "two.txt", "title": "", "text": "Arbor"}]
        self.assertEqual(answer.model_result('prefix ```json\n{"answer":"Fable","source":1,"confidence":2}\n```',
                                             passages)["confidence"], 1)
        self.assertIsNone(answer.model_result('{"answer":"other","source":1}', passages))
        self.assertEqual(answer.model_result('{"answer":"","source":null}', passages), EMPTY)
        self.assertEqual(answer.model_result('{"answer":"Fable Arbor","source":[1,2]}', passages)["citations"],
                         ["one.txt", "two.txt"])
        self.assertEqual(normalise(' " Some   Value. " '), "Some Value")
        self.assertEqual(first_object('noise {invalid} {"answer":"x"}'), {"answer": "x"})
        self.assertEqual(result("x", ["p"], float("nan"))["confidence"], 0)

    def test_crawl_page_limit_and_link_ranking(self):
        visited = []
        def fake_fetch(url, deadline):
            visited.append(url)
            links = [("http://127.0.0.1/" + str(i), "irrelevant") for i in range(30)]
            links.insert(20, ("http://127.0.0.1/prices", QUESTION))
            return {"source": url, "title": "", "text": "", "links": links}
        with mock.patch.object(retrieval, "fetch", fake_fetch):
            retrieval.crawl("http://127.0.0.1/", QUESTION, time.monotonic() + 3)
        self.assertEqual(len(visited), 12)
        self.assertEqual(visited[1], "http://127.0.0.1/prices")

    def test_http_timeout_and_byte_cap(self):
        class Response:
            def __init__(self):
                import email.message
                self.headers = email.message.Message()
                self.headers["Content-Type"] = "text/plain; charset=utf-8"
                self.total = 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def geturl(self):
                return "http://127.0.0.1/capped"
            def read1(self, size):
                self.total += size
                return b"x" * size
        response = Response()
        with mock.patch("urllib.request.urlopen", return_value=response) as opening:
            page = retrieval.fetch("http://127.0.0.1/capped", time.monotonic() + 10)
        self.assertLessEqual(opening.call_args.kwargs["timeout"], 4)
        self.assertEqual(response.total, 1_500_000)
        self.assertEqual(len(page["text"]), 1_500_000)

    def test_fixture_values_absent_from_all_app_files(self):
        # Byte scan is the stdlib equivalent of grep, also covering generated bytecode.
        for path in APP.rglob("*"):
            if path.is_file():
                raw = path.read_bytes()
                for fixture in FIXTURE_VALUES:
                    self.assertNotIn(fixture.encode(), raw, str(path))

    def test_gpu_contract_static_only_and_requirements(self):
        source = (APP / "m4agent" / "backend.py").read_text()
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.assertNotIn("torch", ast.unparse(node))
                self.assertNotIn("transformers", ast.unparse(node))
        self.assertIn('.to("cuda")', source)
        self.assertIn("torch.bfloat16", source)
        self.assertIn("AutoModelForCausalLM", source)
        self.assertNotIn("AutoProcessor", source)
        self.assertNotIn("torchvision", source)
        self.assertEqual((APP / "requirements.txt").read_text().strip(), "transformers")


if __name__ == "__main__":
    unittest.main()
