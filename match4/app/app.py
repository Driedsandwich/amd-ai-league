#!/usr/bin/env python3
"""Harness entry point: publish first, run bounded work, print the saved result."""
import time

STARTED = time.monotonic()

import json
import multiprocessing
import os
import sys
from pathlib import Path

from m4agent.cli import jobs_from_args, parse_args, recover_output
from m4agent.common import atomic_json, result


def process_started():
    # Include interpreter startup only when proc and monotonic clocks agree.
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        proc_start = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        return proc_start if 0 <= STARTED - proc_start <= 5 else STARTED
    except (OSError, ValueError, IndexError):
        return STARTED


def answer_jobs(jobs, deadline):
    from m4agent import client
    from m4agent.answer import fallback, model_result, prompt_for
    from m4agent.retrieval import gather

    client.ensure_server()
    for position, job in enumerate(jobs):
        # Batch mode shares the process deadline; every item already has a placeholder.
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            break
        end = min(deadline, time.monotonic() + remaining / (len(jobs) - position))
        if not job["query"]:
            continue
        try:
            passages = gather(job["query"], job["url"], job["corpus"],
                              min(end - 1, time.monotonic() + 12))
            answer, options = fallback(job["query"], passages)
            atomic_json(job["output"], answer)
            if not passages:
                continue
            # Cold startup gets at most 250 ms; --index is the load/warmup phase.
            if not os.path.exists(client.socket_path()):
                client.wait_ready(min(end - 1, time.monotonic() + 0.25))
            timeout = end - time.monotonic() - 1
            response = client.call({"op": "generate", "prompt": prompt_for(job["query"], passages, options),
                                    "max_new": 64, "max_time": max(0, timeout),
                                    "deadline": end - 1}, timeout)
            if response and isinstance(response.get("text"), str):
                improved = model_result(response["text"], passages)
                if improved is not None:
                    atomic_json(job["output"], improved)
        except Exception as exc:
            print(f"query {job['id']}: {type(exc).__name__}: {exc}", file=sys.stderr)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        return run(argv)
    except Exception as exc:
        print(f"input/run error: {type(exc).__name__}: {exc}", file=sys.stderr)
        saved = result()
        atomic_json(recover_output(argv), saved)
        print(json.dumps(saved, ensure_ascii=False, separators=(",", ":")), flush=True)
        return 0


def run(argv):
    args = parse_args(argv)
    if args.index:
        from m4agent import client
        from m4agent.retrieval import build_index
        client.ensure_server()
        try:
            count = build_index(args.index)
            print(f"indexed {count} files", file=sys.stderr)
        except Exception as exc:
            print(f"index error: {exc}", file=sys.stderr)
        # Eight minutes for readiness, also leaving ten seconds of the startup budget.
        ready = client.wait_ready(min(process_started() + 590, time.monotonic() + 480))
        if not ready:
            print("model server not ready; query fallback remains available", file=sys.stderr)
        return 0

    deadline = process_started() + 24
    jobs = jobs_from_args(args)
    for job in jobs:
        atomic_json(job["output"], result())
    if jobs and time.monotonic() < deadline - 0.5:
        worker = multiprocessing.get_context("fork").Process(target=answer_jobs, args=(jobs, deadline))
        worker.start()
        try:
            # Reserve time for terminating stalled DNS/HTTP/model work and stdout.
            worker.join(max(0, deadline - time.monotonic() - 0.4))
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(0.1)
            if worker.is_alive():
                worker.kill()
                worker.join(0.1)
            worker.close()
    for job in jobs:
        try:
            saved = json.loads(Path(job["output"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = result()
            atomic_json(job["output"], saved)
        print(json.dumps(saved, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
