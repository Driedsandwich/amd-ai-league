#!/usr/bin/env python3
"""Harness entry point: publish first, run bounded work, print the saved result."""
import time

STARTED = time.monotonic()

import json
import multiprocessing
import os
import sys
from collections import Counter
from pathlib import Path

from m4agent.cli import jobs_from_args, parse_args, recover_jobs
from m4agent.common import atomic_json, result


def process_started():
    # Include interpreter startup only when proc and monotonic clocks agree.
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        proc_start = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        return proc_start if 0 <= STARTED - proc_start <= 5 else STARTED
    except (OSError, ValueError, IndexError):
        return STARTED


def publish(job, saved):
    for path in job["outputs"]:
        try:
            atomic_json(path, saved)
        except (OSError, ValueError) as exc:
            print(f"output error: {exc}", file=sys.stderr)


def print_result(saved):
    print(json.dumps(saved, ensure_ascii=False, separators=(",", ":")), flush=True)


def read_result(job, path_counts):
    # In a batch, prefer an item-specific file over a shared alias.
    for path in sorted(job["outputs"], key=path_counts.get):
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(saved, dict) and saved.keys() == result().keys()
                and isinstance(saved["answer"], str) and isinstance(saved["citations"], list)
                and all(isinstance(citation, str) for citation in saved["citations"])
                and type(saved["confidence"]) is float):
            return saved
    saved = result()
    publish(job, saved)
    return saved


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
        if job["error"] or not job["query"]:
            continue
        try:
            passages = gather(job["query"], job["url"], job["corpus"],
                              min(end - 1, time.monotonic() + 12))
            answer, options = fallback(job["query"], passages)
            publish(job, answer)
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
                    publish(job, improved)
        except Exception as exc:
            print(f"query {job['id']}: {type(exc).__name__}: {exc}", file=sys.stderr)


def work_jobs(jobs, deadline, args):
    if os.environ.get("X052_DIAG") == "1":
        from m4agent.diagnostic import claim_invocation, network_available, diagnostic_result

        if claim_invocation():
            net = network_available()
            for job in jobs:
                if not job["error"]:
                    publish(job, diagnostic_result(args, job, net))
            return
    answer_jobs(jobs, deadline)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    jobs = []
    try:
        args = parse_args(argv)
        if args.help:
            print("usage: app.py [--query QUESTION | --input PATH | --index FOLDER] "
                  "[--query-id ID] [--url URL] [--corpus FOLDER] [--output-dir DIRECTORY]",
                  file=sys.stderr)
            return 0
        if args.index and not args.missing:
            return run_index(args.index)
        jobs = jobs_from_args(args)
        if args.missing:
            raise ValueError(f"missing value for {', '.join(args.missing)}")
        run_queries(args, jobs)
    except Exception as exc:
        print(f"input/run error: {type(exc).__name__}: {exc}", file=sys.stderr)
        saved = result()
        try:
            jobs = jobs or recover_jobs(argv)
        except Exception as recovery_error:
            print(f"output recovery error: {recovery_error}", file=sys.stderr)
        for job in jobs:
            publish(job, saved)
        for _ in jobs or [None]:
            print_result(saved)
    return 0


def run_index(directory):
    from m4agent import client
    from m4agent.retrieval import build_index
    client.ensure_server()
    try:
        count = build_index(directory)
        print(f"indexed {count} files", file=sys.stderr)
    except Exception as exc:
        print(f"index error: {exc}", file=sys.stderr)
    # Eight minutes for readiness, also leaving ten seconds of the startup budget.
    ready = client.wait_ready(min(process_started() + 590, time.monotonic() + 480))
    if not ready:
        print("model server not ready; query fallback remains available", file=sys.stderr)
    return 0


def run_queries(args, jobs):
    deadline = process_started() + 24
    for job in jobs:
        publish(job, result())
        if job["error"]:
            print(job["error"], file=sys.stderr)
    if jobs and time.monotonic() < deadline - 0.5:
        worker = multiprocessing.get_context("fork").Process(target=work_jobs, args=(jobs, deadline, args))
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
    path_counts = Counter(path for job in jobs for path in job["outputs"])
    for job in jobs:
        print_result(read_result(job, path_counts))


if __name__ == "__main__":
    raise SystemExit(main())
