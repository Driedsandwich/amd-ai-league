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

Question aliases: `--query`, `--question`, `--task`, `--prompt`, `--q`, or a
positional question. A single positional JSON object accepts the same keys
without the leading dashes.

ID aliases: `--query-id`, `--query_id`, `--id`, `--qid`, `--task-id`, `--task_id`.
Missing ID becomes `query`. Normal harness IDs are preserved; path separators
and other unsafe filename characters become underscores.

Start aliases: `--url`, `--start-url`, `--start_url`, `--website`, `--site`,
`--base-url`, `--base_url`. HTTP(S), local folders, local files and `file://`
paths are supported.

Folder aliases: `--corpus`, `--sites`, `--pages`, `--web-root`, `--web_root`,
`--root`. Explicit corpus takes precedence over the start page. A missing
local path still stays offline. Unknown long flags and their directly following
value are ignored; abbreviation is off. A question flag suppresses all positional
text, including when its value is empty.

`--index <folder>` parses HTML, HTM, TXT, MD and JSON files, catches per-file
failures, writes a compact JSON index under `/app/index/`, starts the resident
server, waits up to eight minutes for readiness, and exits zero. Indexes are
keyed by the resolved corpus root and reused as snapshots; rerun `--index` if
the corpus changes. Symlinks outside the corpus root are skipped. Files are
bounded to 1.5 MB of text each, including offline files.

`--input <file.json|file.jsonl>` accepts a list or JSON-lines objects with
`id|query_id`, `query|question`, optional `url` and `corpus`. Batch items share
the invocation's 24-second budget and get proportional remaining time.
Unique IDs should be supplied; repeated/missing IDs address the same output file.

`--output-dir` defaults to `/app/output`. Each question immediately gets
`<id>_output.json` with an empty answer, empty citations and zero confidence.
Every update uses a temporary file chmodded to 0644 before `os.replace`, so
other UIDs can read it. Argument errors and other top-level exceptions publish
an empty result and exit zero, recovering the ID and output directory from argv
(including positional JSON IDs); the default ID is `query`.
Final JSON is printed as
one stdout line per item, matching the saved output. Additional `sources` and
`source_url` fields mirror citations. Diagnostics go to stderr.

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
answer. Cold model startup is not awaited beyond 250 ms during a question.
Production startup should therefore complete `--index` before questions.

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
python3 -m unittest discover -s mc4/tests -v
```

Tests use stdlib unittest, invented fixtures and a 127.0.0.1 HTTP server.
Subprocess guards reject heavy model imports and outbound DNS/TCP; offline
tests additionally forbid loopback TCP and patch network entry points.
Runtime files and test sockets live under `mc4/` and are cleaned up. The suite
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
