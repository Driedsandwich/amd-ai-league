# Match 4 box

This is a best-guess Match 4 implementation, not a verified Match 4 contract.
The only official reference is the published Mini-Challenge 3 contract; the resident service
follows the Match 2 resident-server pattern in this repository. No sample answers are embedded in application code.

## Files and image layout

Copy the contents of `app/` to `/app/` in a tar layer. The mandated unchanged
base is `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`.
Assemble layers with crane outside this implementation; no Docker build or
package installation was performed here. `app/requirements.txt` lists only
`transformers`. Its dependencies must also be supplied in `/opt/x052/site/`,
without replacing the base image's ROCm torch or other preinstalled packages.

Place text-only Qwen3 weights and tokenizer files in
`/models/Qwen3-4B-Instruct-2507/`. Image configuration:

```text
PYTHONPATH=/opt/x052/site
X052_MODEL_DIR=/models/Qwen3-4B-Instruct-2507
X052_BACKEND=transformers
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
CMD ["python3", "/app/supervise.py"]
```

`supervise.py` holds an advisory lock, starts `server.py`, allows five restarts,
then stays alive. The server has a separate lock, loads once, warms up with a
short prompt followed by about 6,000 characters and eight new tokens,
and listens on `/tmp/x052_m4.sock`. One newline-delimited JSON message per
connection supports `ping` and `generate`; generation uses at most 64 new tokens.
The client includes an absolute monotonic deadline so queued requests expire.
Only the model server imports torch/transformers. It uses bf16,
`AutoTokenizer`, `AutoModelForCausalLM`, chat templates, greedy decoding, and
explicit `.to("cuda")`. Missing GPU raises an error; there is no CPU model path.
Loading passes `dtype=torch.bfloat16`, retrying with `torch_dtype` only on
`TypeError`. Generation disables sampling and clears temperature/top-p/top-k.
Load errors are recorded in `<socket>.status.json`. A stub backend is explicitly
selected for CPU tests; its sleep deliberately ignores generation time limits.

## CLI and outputs

`main.py`, `agent.py`, `run.py`, `solve.py`, `predict.py`, and `inference.py`
all call the same entry point as `app.py`.

Questions accept `--query`, `--question`, `--task`, `--prompt`, `--q`, or
positional text. Inline JSON and input files also accept `request`,
`instruction`, `find`, and `text`. Start URLs accept `--url`, `--start-url`,
`--start_url`, `--website`, `--site`, `--base-url`, and `--base_url`, plus
`link` in JSON. If needed, the longest unknown-option value of at least three
words supplies the question, and the first unknown-option value beginning with
`http://`, `https://`, or `file://` supplies the URL. Corpus aliases are
`--corpus`, `--sites`, `--pages`, `--web-root`, `--web_root`, and `--root`.
An explicit corpus takes precedence over the start page. Local paths stay offline.

Input flags are `--input`, `--input-file`, `--input-json`, `--request`,
`--query-file`, `--question-file`, `--task-file`, `--find`, and `--find-file`.
An existing file passed positionally or as an unknown option's value is also an
input. Files accept a JSON object, an array of objects, or JSON Lines. A
directory passed to an input flag supplies one item per `*.json` file in
filename order, using each file's stem as its ID. Batch items share the
invocation's 24-second budget and get proportional remaining time.

For each item, IDs come from the existing ID flags, other long flags containing
`id`, the JSON keys `id`, `query_id`, `find_id`, `task_id`, `qid`, `item_id`,
`request_id`, and `name`, then matching command-line words or flag values such
as `find_01`, and the environment variables `QUERY_ID`, `FIND_ID`, `TASK_ID`,
and `ITEM_ID`. A single input file supplies its stem when its object has no ID
key with a usable value. IDs are made safe for filenames and deduplicated.
When no candidates remain, the ID is `query`.

Each ID receives identical `<id>_output.json` and `<id>.json` files under
`/app/output` or the given `--output-dir`. Additional directories may be supplied with `--output-dir`,
`--out-dir`, `--output_dir`, `--outdir`, or `--results-dir`. The flags
`--output`, `--output-file`, `--out`, `--output-path`, `--result`, or
`--result-file` add an exact `.json` file or another output directory. A shared
destination in a batch can be overwritten by later items.

All destinations get an empty placeholder before retrieval or inference.
Writes use atomic replacement and mode 0644. The final result is also emitted
as one stdout line per item. Files and stdout carry exactly `answer` as a
string, `citations` as an array of strings, and `confidence` as a number. Input
and top-level errors publish empty results and exit zero.

With `X052_DIAG=1`, a lock-protected counter at `/tmp/x052_m4_diag.count`
reserves diagnostics for the first two question invocations. Indexing skips
the counter. Each diagnostic answer starts with `DIAG` and records sorted
identifier-like long option names and input keys, the positional argument count,
the start URL scheme, whether a corpus was supplied, and the result of a
one-second TCP probe to `1.1.1.1:443`. Diagnostics use `["diag"]` for citations
and zero confidence. Question text, file contents, full URLs, and argument
values are omitted apart from the URL scheme. Subsequent calls answer normally.

`--index <folder>` parses HTML, HTM, TXT, MD and JSON files, catches per-file
failures, writes a compact JSON index under `/app/index/`, starts the resident
server, waits up to eight minutes for readiness, and exits zero. Indexes are
keyed by the resolved corpus root and reused as snapshots. Rerun `--index` if
the corpus changes. Symlinks outside the corpus root are skipped. Files are
bounded to 1.5 MB of text each, including offline files.

## Retrieval, answers and time

A Linux parent process uses `/proc/self/stat` process start only when it is
between zero and five seconds before the module's monotonic start time.
Otherwise it uses the module time to tolerate different clock origins.
It writes placeholders before spawning a query worker and stops that worker
by 23.8 seconds, reserving the rest of the 24-second deadline for final output.
This also bounds stalled DNS, socket reads, or inference independent of Python
exceptions. It cannot guarantee timing during OS suspension or blocked output
storage/stdout. Huge input batches are not a tested harness shape.

Retrieval gets at most 12 seconds and at most 12 pages. HTTP requests have
timeouts no greater than four seconds, absolute read deadlines and a 1.5 MB
body cap. Each fetch catches all ordinary exceptions, including malformed URLs
and incomplete reads, so a failed fetch does not discard other crawl results.
Same-site links and question overlap rank a breadth-first crawl.
Without any URL/folder, optional keyless DuckDuckGo search supplies result URLs;
network errors produce an empty answer. No search runs for offline inputs.
HTML cleaning drops scripts, styles, navigation and footers, preserves titles,
canonical URLs, and table rows as `cell | cell`. Offline citations stay relative
to the corpus even when HTML contains a canonical link.

BM25-style scoring selects offline files and eight overlapping passages of
roughly 800 characters. Type-specific regexes extract candidate values for
prices, numbers, counts, dates, years, versions, email, phone, URL, percentages,
yes/no, and explicit name/other table or label values. Ambiguous ties refuse.
The rule answer is saved before inference. A model request gets the remaining
time minus one second; failure or malformed/ungrounded output retains the rule
answer. After writing placeholders, a question with no server socket starts the
resident supervisor in the background and waits up to 250 ms total for readiness.

Prompts contain numbered passages, source paths/URLs and candidates. Model
output is parsed from the first JSON object; answers must occur in the selected
evidence, with one source or two when requested by the model. This grounding
check prevents invented values but cannot prove semantic correctness or that
both citations are necessary. Explicit model refusal writes empty answer and
citations. Normalization strips surrounding quotes/trailing periods and collapses
spaces while retaining units contained in the selected value.

## Validation and limits

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 X052_BACKEND=stub python3 -m unittest discover -s match4/tests -v
```

Tests use stdlib unittest, invented fixtures and a 127.0.0.1 HTTP server.
Subprocess guards reject heavy model imports and outbound DNS/TCP; offline
tests additionally forbid loopback TCP and patch network entry points.
Runtime files and test sockets live under `match4/` and are cleaned up. The suite
covers CLI aliases, JSON/JSONL inputs, index startup/reuse, multihop retrieval,
citations, resident reuse, fallbacks, external kill survival, the parent watchdog,
an actually sleeping stub, fetch caps, HTML cleaning and fixture leakage.
Review regressions also cover malformed links/read failures, clock disagreement,
error output recovery, output permissions, unknown-option values, and mocked
backend loading, generation parameters and both warmup calls.

Untested: actual GPU/ROCm loading, Qwen3 weights/tokenizer compatibility,
generation speed and quality, VRAM usage/floor, startup time with real weights,
crane image assembly/layer identity/image size, public HTTP(S)/DuckDuckGo,
and the unpublished Match 4 harness. No weights or packages were installed.
Static GPU-code checks and CPU stub tests do not validate any of those paths.
