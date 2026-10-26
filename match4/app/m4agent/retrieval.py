"""Bounded stdlib retrieval, HTML cleaning, offline indexing and passage ranking."""
import hashlib
import heapq
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from .common import atomic_json

MAX_BYTES = 1_500_000
MAX_PAGES = 12
EXTENSIONS = {".html", ".htm", ".txt", ".md", ".json"}
STOP = set("a an the is are was were be of for to in on at and or what which who when where how much many does do did please tell me give it its this that".split())


def terms(text):
    return [t for t in re.findall(r"[\w]+", text.lower()) if t not in STOP]


class Cleaner(HTMLParser):
    DROP = {"script", "style", "nav", "footer", "noscript", "template"}
    BLOCK = {"p", "div", "section", "article", "h1", "h2", "h3", "li", "br", "tr", "table"}

    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base, self.canonical = base, base
        self.parts, self.title, self.links, self.hidden = [], [], [], []
        self.in_title = False
        self.anchor = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self.DROP:
            self.hidden.append(tag)
        if self.hidden:
            return
        if tag == "title":
            self.in_title = True
        if tag == "link" and "canonical" in attrs.get("rel", "").lower().split():
            canonical = urllib.parse.urljoin(self.base, attrs.get("href", ""))
            if urllib.parse.urlsplit(canonical).scheme in {"http", "https"}:
                self.canonical = canonical
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag in {"td", "th"}:
            self.parts.append(" | ")
        if tag == "a":
            self.anchor = [urllib.parse.urljoin(self.base, attrs.get("href", "")), []]

    def handle_endtag(self, tag):
        if self.hidden:
            if tag in self.hidden:
                self.hidden = self.hidden[:self.hidden.index(tag)]
            return
        if tag == "title":
            self.in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "a" and self.anchor is not None:
            if len(self.links) < 256:
                self.links.append((self.anchor[0], " ".join(self.anchor[1])))
            self.anchor = None

    def handle_data(self, data):
        if self.hidden:
            return
        if self.in_title:
            self.title.append(data)
        else:
            self.parts.append(data)
        if self.anchor is not None:
            self.anchor[1].append(data)


def page_from_text(raw, source, html=False):
    if html:
        parser = Cleaner(source)
        parser.feed(raw)
        title = re.sub(r"\s+", " ", " ".join(parser.title)).strip()
        lines = [re.sub(r"[\t\r\f\v ]+", " ", line).strip(" |")
                 for line in "".join(parser.parts).split("\n")]
        text = "\n".join(line for line in lines if line)
        return {"source": source, "canonical": parser.canonical, "title": title,
                "text": text, "links": parser.links}
    return {"source": source, "canonical": source, "title": Path(source).stem,
            "text": raw, "links": []}


def http_url(value):
    return urllib.parse.urlsplit(str(value)).scheme in {"http", "https"}


def fetch(url, deadline):
    try:
        end = min(deadline, time.monotonic() + 4)
        if not http_url(url) or end <= time.monotonic():
            return None
        request = urllib.request.Request(url, headers={"User-Agent": "M4Retrieval/1.0", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=max(0.001, end - time.monotonic())) as response:
            chunks, size = [], 0
            while size < MAX_BYTES and time.monotonic() < end:
                # HTTPResponse.read1 performs a single read; enforce the total fetch budget.
                try:
                    response.fp.raw._sock.settimeout(max(0.001, end - time.monotonic()))
                except AttributeError:
                    pass
                chunk = response.read1(min(65536, MAX_BYTES - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            content_type = response.headers.get_content_type()
            encoding = response.headers.get_content_charset() or "utf-8"
            raw = b"".join(chunks).decode(encoding, errors="replace")
            is_html = "html" in content_type or bool(re.search(r"<(?:html|table|body)\b", raw, re.I))
            page = page_from_text(raw, response.geturl(), is_html)
            page["source"] = page["canonical"]
            return page
    except Exception:
        # A bad URL, response or page must not discard the rest of the crawl.
        return None


def crawl(start, question, deadline):
    pages, seen = [], set()
    queue, serial = [], 0
    origin = urllib.parse.urlsplit(start).netloc.lower()
    query = set(terms(question))
    heapq.heappush(queue, (0, 0, serial, start))
    while queue and len(seen) < MAX_PAGES and time.monotonic() < deadline:
        depth, _, _, url = heapq.heappop(queue)
        url = urllib.parse.urldefrag(url)[0]
        if url in seen or not http_url(url):
            continue
        seen.add(url)
        page = fetch(url, deadline)
        if not page:
            continue
        pages.append(page)
        for link, label in page["links"]:
            if not http_url(link) or link in seen:
                continue
            same = urllib.parse.urlsplit(link).netloc.lower() == origin
            score = len(query.intersection(terms(label + " " + urllib.parse.unquote(link)))) + 2 * same
            serial += 1
            heapq.heappush(queue, (depth + 1, -score, serial, link))
    return pages


def search(question, deadline):
    page = fetch("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(question), deadline)
    if not page:
        return []
    pages, seen = [], set()
    for link, _ in page["links"]:
        params = urllib.parse.parse_qs(urllib.parse.urlsplit(link).query)
        link = params.get("uddg", [link])[0]
        if not http_url(link) or "duckduckgo.com" in urllib.parse.urlsplit(link).netloc or link in seen:
            continue
        seen.add(link)
        if len(seen) > MAX_PAGES - 1 or time.monotonic() >= deadline:
            break
        found = fetch(link, deadline)
        if found:
            pages.append(found)
    return pages


def index_path(root):
    key = hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:20]
    return Path(os.environ.get("X052_INDEX_DIR", "/app/index")) / (key + ".json")


def read_folder(root, deadline=float("inf")):
    root = Path(root).resolve()
    base = root if root.is_dir() else root.parent
    paths = (Path(directory) / name for directory, _, names in os.walk(root)
             for name in sorted(names)) if root.is_dir() else iter([root])
    for path in paths:
        if time.monotonic() >= deadline:
            break
        if path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            # Never follow a corpus symlink out of its root.
            path.resolve().relative_to(base)
            with path.open("rb") as stream:
                raw = stream.read(MAX_BYTES).decode("utf-8", errors="replace")
            yield page_from_text(raw, path.relative_to(base).as_posix(), path.suffix.lower() in {".html", ".htm"})
        except (OSError, ValueError):
            continue


def build_index(root):
    pages = list(read_folder(root))
    atomic_json(index_path(root), {"version": 1, "root": str(Path(root).resolve()), "pages": pages})
    return len(pages)


def offline_pages(root, deadline):
    try:
        data = json.loads(index_path(root).read_text(encoding="utf-8"))
        if data["version"] == 1 and data["root"] == str(Path(root).resolve()):
            return data["pages"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return list(read_folder(root, deadline))


def scores(question, texts):
    query = set(terms(question))
    counters = [Counter(terms(text)) for text in texts]
    average = sum(sum(c.values()) for c in counters) / max(1, len(counters)) or 1
    df = {t: sum(t in c for c in counters) for t in query}
    values = []
    for counter in counters:
        length = sum(counter.values())
        score = 0.0
        for term in query:
            frequency = counter[term]
            idf = math.log(1 + (len(counters) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average))
        values.append(score)
    return values


def passages(pages, question):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(start + 800, len(text))
            if end < len(text):
                boundary = text.rfind("\n", start + 400, end)
                if boundary != -1:
                    end = boundary
            chunks.append({"source": page["source"], "title": page["title"],
                           "text": text[start:end]})
            if end == len(text):
                break
            start = max(start + 1, end - 120)
    ranking = scores(question, [p["title"] + " " + p["text"] for p in chunks])
    ranked = sorted(zip(ranking, chunks), key=lambda pair: pair[0], reverse=True)
    return [dict(p, score=score) for score, p in ranked[:8] if score > 0]


def gather(question, start, corpus, deadline):
    # A supplied path is always offline, including nonexistent paths and file:// URLs.
    if corpus:
        pages = offline_pages(corpus, deadline)
    elif start and not http_url(start):
        path = urllib.parse.unquote(urllib.parse.urlsplit(start).path) if start.startswith("file:") else start
        pages = offline_pages(path, deadline)
    elif start:
        pages = crawl(start, question, deadline)
    else:
        pages = search(question, deadline)
    if corpus or (start and not http_url(start)):
        ranking = scores(question, [p["title"] + " " + p["source"] + " " + p["text"] for p in pages])
        pages = [p for _, p in sorted(zip(ranking, pages), key=lambda pair: pair[0], reverse=True)[:12]]
    return passages(pages, question)
