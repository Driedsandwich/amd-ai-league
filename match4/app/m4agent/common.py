"""Small shared helpers; no model or network imports."""
import json
import math
import os
import re
import tempfile
from pathlib import Path


def normalise(value):
    return re.sub(r"\s+", " ", str(value)).strip().strip('"\'“”‘’').strip().rstrip(".").strip()


def result(answer="", citations=(), confidence=0.0):
    answer = normalise(answer)
    citations = list(dict.fromkeys(str(c) for c in citations if c))[:2] if answer else []
    if not citations:
        answer = ""
    try:
        confidence = float(confidence)
        confidence = min(1.0, max(0.0, confidence)) if math.isfinite(confidence) else 0.0
    except (ValueError, TypeError):
        confidence = 0.0
    return {"answer": answer, "citations": citations,
            "confidence": confidence if answer else 0.0,
            "sources": citations, "source_url": citations[0] if citations else ""}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def first_object(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
    return None
