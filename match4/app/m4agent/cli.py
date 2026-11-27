"""Discover harness arguments and plan an output for every plausible item ID."""
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

QUERY_FLAGS = ("--query", "--question", "--task", "--prompt", "--q")
ID_FLAGS = ("--query-id", "--query_id", "--id", "--qid", "--task-id", "--task_id")
URL_FLAGS = ("--url", "--start-url", "--start_url", "--website", "--site", "--base-url", "--base_url")
CORPUS_FLAGS = ("--corpus", "--sites", "--pages", "--web-root", "--web_root", "--root")
INPUT_FLAGS = ("--input", "--input-file", "--input-json", "--request", "--query-file",
               "--question-file", "--task-file", "--find", "--find-file")
OUTPUT_DIR_FLAGS = ("--output-dir", "--out-dir", "--output_dir", "--outdir", "--results-dir")
OUTPUT_PATH_FLAGS = ("--output", "--output-file", "--out", "--output-path", "--result", "--result-file")
VALUE_FLAGS = set(QUERY_FLAGS + ID_FLAGS + URL_FLAGS + CORPUS_FLAGS + INPUT_FLAGS
                  + OUTPUT_DIR_FLAGS + OUTPUT_PATH_FLAGS + ("--index",))
ID_KEYS = ("id", "query_id", "find_id", "task_id", "qid", "item_id", "request_id", "name")
QUERY_KEYS = tuple(flag[2:] for flag in QUERY_FLAGS) + ("request", "instruction", "find", "text")
URL_KEYS = tuple(flag[2:] for flag in URL_FLAGS) + ("link",)
CORPUS_KEYS = tuple(flag[2:] for flag in CORPUS_FLAGS)
ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*_[0-9]{1,4}")
ID_ENV = ("QUERY_ID", "FIND_ID", "TASK_ID", "ITEM_ID")


def is_positional_word(word):
    return not word.startswith("-") or word == "-" or re.fullmatch(r"-\d+(?:\.\d+)?", word)


def scan_args(argv):
    options, positional = [], []
    words = iter(argv)
    token = next(words, None)
    while token is not None:
        if token == "--":
            positional.extend(words)
            break
        flag, equals, value = token.partition("=")
        if token in ("-h", "--help"):
            options.append((token, None))
            token = next(words, None)
        elif flag.startswith("--"):
            token = next(words, None)
            if equals:
                options.append((flag, value))
            elif token is not None and (not token.startswith("-")
                                        or (flag in VALUE_FLAGS and is_positional_word(token))):
                options.append((flag, token))
                token = next(words, None)
            else:
                options.append((flag, None))
        else:
            if is_positional_word(token):
                positional.append(token)
            token = next(words, None)
    return options, positional


def last_value(options, flags):
    return next((value for flag, value in reversed(options) if flag in flags and value is not None), None)


def item_value(item, keys):
    return next((str(item[key]) for key in keys if item.get(key)), None)


def is_file(value):
    try:
        return Path(value).is_file()
    except (OSError, ValueError):
        return False


def parse_args(argv=None):
    options, positional = scan_args(sys.argv[1:] if argv is None else argv)
    unknown = [(flag, value) for flag, value in options
               if flag not in VALUE_FLAGS and value is not None]
    files = [Path(value) for flag, value in options if value and flag in INPUT_FLAGS]
    files.extend(Path(value) for _, value in unknown if is_file(value))
    positional_files = {value for value in positional if is_file(value)}
    files.extend(Path(value) for value in positional if value in positional_files)
    text = [value for value in positional if value not in positional_files]
    raw = " ".join(text)
    inline = {}
    if raw.lstrip().startswith("{"):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                inline = data
                raw = ""
        except ValueError:
            pass

    query_flag = last_value(options, QUERY_FLAGS)
    query = query_flag
    if query is None:
        query = item_value(inline, QUERY_KEYS)
        if query is None and raw and not all(ID_PATTERN.fullmatch(value) for value in text):
            query = raw
    unknown_query = max((value for _, value in unknown if len(value.split()) >= 3),
                        key=len, default=None)
    unknown_url = next((value for _, value in unknown
                        if value.startswith(("http://", "https://", "file://"))), None)
    url = last_value(options, URL_FLAGS) or item_value(inline, URL_KEYS)
    corpus = last_value(options, CORPUS_FLAGS) or item_value(inline, CORPUS_KEYS)

    return SimpleNamespace(
        query=query or unknown_query or query, url=url or unknown_url, corpus=corpus,
        query_id=last_value(options, ID_FLAGS) or item_value(inline, ID_KEYS),
        positional=[] if query_flag is not None else text, positional_count=len(positional),
        inline=inline, files=list(dict.fromkeys(files)), options=options,
        command_ids=[value for flag, value in options if flag in ID_FLAGS]
        + [value for flag, value in options if flag not in ID_FLAGS and flag.startswith("--")
           and "id" in flag[2:].lower()],
        word_ids=[word for word in positional + [value for _, value in options if value is not None]
                  if ID_PATTERN.fullmatch(word)],
        index=last_value(options, ("--index",)), help=any(flag in ("-h", "--help") for flag, _ in options),
        missing=[flag for flag, value in options if flag in VALUE_FLAGS and value is None],
    )


def read_items(path):
    raw = path.read_text(encoding="utf-8")
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
    return [item for item in items if isinstance(item, dict)]


def input_items(path):
    try:
        if path.is_dir():
            for file in sorted(path.glob("*.json")):
                if file.is_file():
                    try:
                        item = json.loads(file.read_text(encoding="utf-8"))
                        if isinstance(item, dict):
                            yield item, file.stem, None, True
                            continue
                    except (OSError, ValueError):
                        pass
                    yield {}, file.stem, f"cannot read JSON object: {file}", True
            return
        if not path.is_file():
            yield {}, None, f"not an input file: {path}", False
            return
        items = read_items(path)
    except (OSError, ValueError) as exc:
        yield {}, path.stem if is_file(path) else None, str(exc), False
        return
    for item in items:
        yield item, path.stem, None, False
    if not items:
        yield {}, path.stem, f"no JSON objects in {path}", False


def output_locations(options):
    primary_dir = last_value(options, ("--output-dir",)) or "/app/output"
    directories, files = [Path(primary_dir)], []
    for flag, value in options:
        if value and flag in OUTPUT_DIR_FLAGS + OUTPUT_PATH_FLAGS:
            path = Path(value)
            if flag in OUTPUT_PATH_FLAGS and value.endswith(".json") and not path.is_dir():
                files.append(path)
            else:
                directories.append(path)
    return directories, files


def safe_ids(values):
    return list(dict.fromkeys(
        re.sub(r"[^\w.-]", "_", str(value))[:200]
        for value in values if type(value) in (str, int, float) and str(value).strip()
    ))


def jobs_from_args(args):
    directories, files = output_locations(args.options)
    word_and_env_ids = args.word_ids + [os.environ.get(key) for key in ID_ENV]
    items = [item for path in args.files for item in input_items(path)]
    if not items:
        items = [(args.inline, None, "no JSON files in input directories" if args.files else None, False)]

    jobs = []
    for item, stem, error, from_directory in items:
        input_ids = [item.get(key) for key in ID_KEYS]
        if from_directory or not safe_ids(input_ids):
            input_ids = [stem]
        ids = safe_ids(args.command_ids + input_ids + word_and_env_ids) or ["query"]
        outputs = [directory / (item_id + suffix)
                   for suffix in ("_output.json", ".json")
                   for item_id in ids for directory in directories] + files
        file_item = item if args.files else {}
        jobs.append({"query": item_value(file_item, QUERY_KEYS) or args.query or "",
                     "id": ids[0],
                     "url": item_value(file_item, URL_KEYS) or args.url,
                     "corpus": item_value(file_item, CORPUS_KEYS) or args.corpus,
                     "outputs": list(dict.fromkeys(path.absolute() for path in outputs)),
                     "input_keys": list(item),
                     "error": error})
    return jobs


def recover_jobs(argv):
    return jobs_from_args(parse_args(argv))
