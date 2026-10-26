"""Permissive harness argument aliases and JSON inputs."""
import argparse
import json
import re
import sys
from pathlib import Path

QUERY_FLAGS = ("--query", "--question", "--task", "--prompt", "--q")
ID_FLAGS = ("--query-id", "--query_id", "--id", "--qid", "--task-id", "--task_id")
URL_FLAGS = ("--url", "--start-url", "--start_url", "--website", "--site", "--base-url", "--base_url")
CORPUS_FLAGS = ("--corpus", "--sites", "--pages", "--web-root", "--web_root", "--root")
VALUE_FLAGS = set(QUERY_FLAGS + ID_FLAGS + URL_FLAGS + CORPUS_FLAGS +
                  ("--index", "--input", "--output-dir"))


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # Let the entry point publish a refusal instead of argparse exiting with 2.
        raise ValueError(message)


def without_unknown_options(argv):
    cleaned = []
    index = 0
    while index < len(argv):
        token = argv[index]
        flag, equals, _ = token.partition("=")
        if token == "--":
            cleaned.extend(argv[index:])
            break
        if flag.startswith("--") and flag not in VALUE_FLAGS and flag != "--help":
            # Unknown long options may consume one following word, never another flag.
            if not equals and index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                index += 1
        else:
            cleaned.append(token)
            if flag in VALUE_FLAGS and not equals and index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                index += 1
                cleaned.append(argv[index])
        index += 1
    return cleaned


def output_path(query_id, output_dir):
    # Preserve harness IDs, but never permit paths outside the output directory.
    query_id = re.sub(r"[^\w.-]", "_", str(query_id or "query"))[:200] or "query"
    return query_id, str(Path(output_dir) / (query_id + "_output.json"))


def recover_output(argv):
    """Recover output identity even when parsing stopped before reaching the ID."""
    query_id, json_id, output_dir = None, None, "/app/output"
    for index, token in enumerate(argv):
        flag, equals, value = token.partition("=")
        if flag in ID_FLAGS or flag == "--output-dir":
            if not equals:
                value = argv[index + 1] if index + 1 < len(argv) else ""
            if value and not value.startswith("-"):
                if flag in ID_FLAGS:
                    query_id = value
                else:
                    output_dir = value
        elif token.startswith("{"):
            try:
                data = json.loads(token)
                if isinstance(data, dict):
                    json_id = next((data[f[2:]] for f in ID_FLAGS if data.get(f[2:])), json_id)
            except ValueError:
                pass
    return output_path(query_id or json_id, output_dir)[1]


def parse_args(argv=None):
    argv = without_unknown_options(list(sys.argv[1:] if argv is None else argv))
    parser = ArgumentParser(allow_abbrev=False)
    for flags, dest in [(QUERY_FLAGS, "query"), (ID_FLAGS, "query_id"),
                        (URL_FLAGS, "url"), (CORPUS_FLAGS, "corpus")]:
        parser.add_argument(*flags, dest=dest)
    parser.add_argument("--index")
    parser.add_argument("--input")
    parser.add_argument("--output-dir", default="/app/output")
    parser.add_argument("positional", nargs="*")
    args, _ = parser.parse_known_intermixed_args(argv)
    if args.query is not None:
        args.positional = []
    if args.query is None and args.positional:
        raw = " ".join(args.positional)
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if isinstance(data, dict):
            for flags, dest in [(QUERY_FLAGS, "query"), (ID_FLAGS, "query_id"),
                                (URL_FLAGS, "url"), (CORPUS_FLAGS, "corpus")]:
                if getattr(args, dest) is None:
                    value = next((data[f[2:]] for f in flags if f[2:] in data), None)
                    if value is not None:
                        setattr(args, dest, str(value))
        else:
            args.query = raw
    return args


def jobs_from_args(args):
    if args.input:
        raw = Path(args.input).read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
            items = data if isinstance(data, list) else [data]
        except ValueError:
            items = []
            for line in raw.splitlines():
                try:
                    items.append(json.loads(line))
                except ValueError:
                    continue
    else:
        items = [{"query_id": args.query_id, "query": args.query,
                  "url": args.url, "corpus": args.corpus}]
    jobs = []
    for item in items:
        if not isinstance(item, dict):
            continue
        query_id = item.get("query_id") or item.get("id") or "query"
        query_id, output = output_path(query_id, args.output_dir)
        jobs.append({"query": str(item.get("query") or item.get("question") or ""),
                     "id": query_id, "url": item.get("url") or args.url,
                     "corpus": item.get("corpus") or args.corpus,
                     "output": output})
    return jobs
