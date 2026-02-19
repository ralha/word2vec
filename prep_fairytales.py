
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prep_fairytales.py
Collects and preprocesses public-domain fairy-tale texts for Word2Vec training.

Usage examples:
    # From a list of URLs (Gutenberg HTML/TXT)
    python prep_fairytales.py --urls urls.txt --outdir corpus

    # From a local folder with .txt/.html files
    python prep_fairytales.py --indir ./raw_fairytales --outdir corpus

    # With light normalization (lowercase, punctuation spacing) and basic dedup
    python prep_fairytales.py --urls urls.txt --outdir corpus --lower --dedup
"""

import argparse
import html
import io
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

try:
    import requests
except ImportError as e:
    print("This script requires 'requests'. Install via: pip install requests", file=sys.stderr)
    raise e

# -------------------------
# Utilities
# -------------------------

RE_MULTISPACE = re.compile(r"\s+")
# General-purpose sentence splitter: keep . ? ! followed by space/cap or EOL
RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
# Tokenization: words with apostrophes/hyphens + numbers; keeps Unicode letters
RE_TOKENS = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*|\d+", re.UNICODE)

# Project Gutenberg boilerplate sentinels
GUT_START = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
GUT_END   = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK", re.IGNORECASE)

def read_urls_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

def fetch_url(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    # Try to honor encoding if provided
    r.encoding = r.apparent_encoding or r.encoding
    return r.text

def load_local(path: Path) -> str:
    # Read as text; fallback to latin-1 if UTF-8 fails
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")

def strip_html(raw: str) -> str:
    # Very light HTML stripping without BeautifulSoup
    # 1) Remove scripts/styles
    no_script = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    # 2) Replace <br>, <p>, <div> with newlines
    with_breaks = re.sub(r"(?i)</?(br|p|div|h\d|li|ul|ol|blockquote|pre)>", "\n", no_script)
    # 3) Strip other tags
    text = re.sub(r"(?s)<.*?>", " ", with_breaks)
    # 4) Unescape HTML entities
    text = html.unescape(text)
    return text

def normalize_unicode(text: str) -> str:
    # NFKC normalization removes oddities; preserves Unicode letters
    return unicodedata.normalize("NFKC", text)

def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg headers/footers if present.
    We scan for START/END sentinels and keep content between them; otherwise return as-is.
    """
    lines = text.splitlines()
    start_idx, end_idx = None, None
    for i, ln in enumerate(lines):
        if start_idx is None and GUT_START.search(ln):
            start_idx = i + 1
        if GUT_END.search(ln):
            end_idx = i
            break
    if start_idx is not None and end_idx is not None and start_idx < end_idx:
        body = "\n".join(lines[start_idx:end_idx])
    else:
        body = text
    return body

def basic_clean(text: str, lower: bool = False) -> str:
    # De-dup whitespace, normalize quotes/dashes a bit
    text = normalize_unicode(text)
    # Replace fancy quotes/dashes with simple equivalents
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    text = RE_MULTISPACE.sub(" ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    if lower:
        text = text.lower()
    return text.strip()

def basic_clean(text: str, lower: bool = False) -> str:
    text = normalize_unicode(text)
    # 1. Standardize quotes/dashes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    
    # 2. ADD THIS: Pad punctuation with spaces 
    # This turns "king," into "king ," so they aren't glued together
    text = re.sub(r"([.,!?();:])", r" \1 ", text)
    
    text = RE_MULTISPACE.sub(" ", text)
    if lower:
        text = text.lower()
    return text.strip()

import nltk
nltk.download('punkt_tab')

def split_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text)
def tokenize(text: str) -> List[str]:
    return RE_TOKENS.findall(text.lower())

def is_mostly_english(text: str, threshold: float = 0.7) -> bool:
    # Crude language check: ratio of ASCII letters/punctuation to all chars
    if not text:
        return False
    ascii_like = sum(1 for ch in text if ord(ch) < 128)
    ratio = ascii_like / max(1, len(text))
    return ratio >= threshold

def title_from_text(text: str, fallback_id: str) -> str:
    """
    Extract a reasonable title:
    - First non-empty line under 120 chars
    - Otherwise fallback to id
    """
    for ln in text.splitlines():
        ln = ln.strip()
        if 3 <= len(ln) <= 120 and not ln.isupper():
            return ln
    return f"story_{fallback_id}"

def deduplicate_texts(texts: List[str]) -> List[str]:
    # Remove near exact duplicates by normalized hash (length+few slices)
    seen: set = set()
    result = []
    for t in texts:
        key = (len(t), t[:200], t[-200:])
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result

# -------------------------
# Pipeline
# -------------------------

def preprocess_items(
    raw_items: List[Tuple[str, str]],
    lower: bool = False,
    keep_non_english: bool = False,
) -> Tuple[List[Dict], Dict]:
    """
    raw_items: list of (id, raw_text)

    Returns:
        stories (list of dicts: id, title, text, sentences, token_count)
        stats (dict)
    """
    stories = []
    token_counter = Counter()
    removed_non_english = 0

    for sid, raw in raw_items:
        # If looks like HTML, strip tags first
        text = strip_html(raw) if "<" in raw and ">" in raw else raw
        text = strip_gutenberg_boilerplate(text)
        text = basic_clean(text, lower=lower)

        if not text or len(text) < 300:
            continue

        # Optional language gate
        if not keep_non_english and not is_mostly_english(text):
            removed_non_english += 1
            continue

        # Sentence split & tokenization
        sents = split_sentences(text)
        toks = tokenize(text)

        title = title_from_text(text, sid)

        stories.append({
            "id": sid,
            "title": title,
            "text": text,
            "sentences": sents,
            "token_count": len(toks),
        })
        token_counter.update(toks)

    stats = {
        "stories": len(stories),
        "tokens": sum(s["token_count"] for s in stories),
        "vocab": len(token_counter),
        "removed_non_english": removed_non_english,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    return stories, stats

def write_outputs(stories: List[Dict], stats: Dict, outdir: Path):
    out_clean = outdir / "clean"
    out_clean.mkdir(parents=True, exist_ok=True)

    # JSONL (one per line)
    with (out_clean / "stories.jsonl").open("w", encoding="utf-8") as f:
        for s in stories:
            f.write(json.dumps({
                "id": s["id"],
                "title": s["title"],
                "text": s["text"],
            }, ensure_ascii=False) + "\n")

    # Plain concatenated corpus (blank line separator)
    with (out_clean / "corpus.txt").open("w", encoding="utf-8") as f:
        for s in stories:
            f.write(s["text"].strip() + "\n\n")

    # Tokenized (one story per line)
    with (out_clean / "tokens.txt").open("w", encoding="utf-8") as f:
        for s in stories:
            tokens = tokenize(s["text"])
            f.write(" ".join(tokens) + "\n")

    # Stats
    with (out_clean / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Helpful printout
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"✔ Wrote {len(stories)} stories to {out_clean}")


def load_from_urls(urls: List[str]) -> List[Tuple[str, str]]:
    items = []
    for i, url in enumerate(urls, 1):
        try:
            txt = fetch_url(url)
            sid = f"url_{i:04d}"
            items.append((sid, txt))
            print(f"Fetched {url}")
        except Exception as e:
            print(f"Skipping {url}: {e}", file=sys.stderr)
    return items

def load_from_indir(indir: Path) -> List[Tuple[str, str]]:
    items = []
    for p in sorted(indir.glob("**/*")):
        if p.is_dir(): 
            continue
        if p.suffix.lower() not in {".txt", ".html", ".htm"}:
            continue
        try:
            txt = load_local(p)
            items.append((p.stem, txt))
            print(f"Loaded {p}")
        except Exception as e:
            print(f"Skipping {p}: {e}", file=sys.stderr)
    return items

def main():
    ap = argparse.ArgumentParser(description="Preprocess fairy tales for Word2Vec.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--urls", type=str, help="Path to a text file with one URL per line")
    src.add_argument("--indir", type=str, help="Directory containing raw .txt/.html files")
    ap.add_argument("--outdir", type=str, default="corpus", help="Output directory")
    ap.add_argument("--lower", action="store_true", help="Lowercase the text")
    ap.add_argument("--keep-non-english", action="store_true", help="Keep non-English content")
    ap.add_argument("--dedup", action="store_true", help="Remove near duplicates")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.urls:
        urls = read_urls_file(Path(args.urls))
        raw_items = load_from_urls(urls)
    else:
        raw_items = load_from_indir(Path(args.indir))

    if args.dedup:
        raw_texts = deduplicate_texts([t for _, t in raw_items])
        # Reassign ids after dedup to maintain mapping
        raw_items = [(f"doc_{i:04d}", t) for i, t in enumerate(raw_texts, 1)]

    stories, stats = preprocess_items(
        raw_items,
        lower=args.lower,
        keep_non_english=args.keep_non_english
    )
    write_outputs(stories, stats, outdir)

if __name__ == "__main__":
    main()
