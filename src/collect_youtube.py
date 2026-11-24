"""Collect Romanized Punjabi-English comments via the YouTube Data API v3.

Compliant (official API, not scraping) and privacy-preserving: we store ONLY the
comment text + video id — never usernames, author ids, or profile data. @mentions
are stripped before saving.

Setup (free):
  1. console.cloud.google.com -> new project -> enable "YouTube Data API v3".
  2. Create an API key.
  3. export YOUTUBE_API_KEY="your_key"      (or pass --api-key)

Usage:
  # by explicit video ids (best: pick Punjabi news/vlog/music videos with many comments)
  python src/collect_youtube.py --videos dQw4w9WgXcQ,AbCdEf12345 --max-per-video 500

  # by search query (finds Punjabi videos, then pulls their comments)
  python src/collect_youtube.py --query "punjabi news debate" --max-videos 15 --max-per-video 300

Output: data/raw/youtube_comments.csv  (columns: text, video_id, collected_at)
Then:   python src/make_annotation_sheet.py --input data/raw/youtube_comments.csv

Note: the API has a daily quota (default 10,000 units). search.list costs 100
units/call; commentThreads.list costs 1 unit/call — so prefer --videos when you can.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

API_ROOT = "https://www.googleapis.com/youtube/v3"
MENTION_RE = re.compile(r"@[\w.-]+")
URL_RE = re.compile(r"https?://\S+")

# Script ranges we DON'T want in a Romanized set
DEVANAGARI = (0x0900, 0x097F)
GURMUKHI = (0x0A00, 0x0A7F)

# Romanized Punjabi / Hinglish function-word markers (heuristic language filter).
PUNJABI_MARKERS = {
    "nahi", "nai", "nhi", "hai", "haiga", "te", "da", "di", "de", "nu", "tu",
    "tusi", "tuhada", "assi", "ehna", "ohna", "kudi", "kudiyan", "munda",
    "veere", "veer", "paji", "bhaji", "changa", "vadiya", "bakwaas", "kiven",
    "ki", "kyu", "sanu", "tuanu", "menu", "oye", "yaar", "banda", "gall",
    "karda", "karde", "hunde", "rehna", "jaa", "aa", "chak", "putt",
}


def _script_frac(text: str, lo: int, hi: int) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)


def looks_romanized_punjabi(text: str, *, min_markers: int = 1) -> bool:
    """Keep Latin-script comments that carry Punjabi/Hinglish markers."""
    if _script_frac(text, *DEVANAGARI) > 0.2 or _script_frac(text, *GURMUKHI) > 0.2:
        return False  # native-script, not Romanized
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & PUNJABI_MARKERS) >= min_markers


def _clean_for_storage(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)          # strip handles (privacy)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _api_get(endpoint: str, params: dict, api_key: str) -> dict:
    params = {**params, "key": api_key}
    url = f"{API_ROOT}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))


def search_video_ids(query: str, max_videos: int, api_key: str) -> list[str]:
    ids, token = [], None
    while len(ids) < max_videos:
        params = {
            "part": "id", "q": query, "type": "video",
            "maxResults": min(50, max_videos - len(ids)),
            "relevanceLanguage": "pa", "order": "relevance",
        }
        if token:
            params["pageToken"] = token
        data = _api_get("search", params, api_key)
        ids += [it["id"]["videoId"] for it in data.get("items", []) if "videoId" in it.get("id", {})]
        token = data.get("nextPageToken")
        if not token:
            break
    print(f"  found {len(ids)} videos for query {query!r}")
    return ids


def fetch_comments(video_id: str, max_comments: int, api_key: str) -> list[str]:
    out, token = [], None
    while len(out) < max_comments:
        params = {
            "part": "snippet", "videoId": video_id,
            "maxResults": 100, "textFormat": "plainText", "order": "relevance",
        }
        if token:
            params["pageToken"] = token
        try:
            data = _api_get("commentThreads", params, api_key)
        except Exception as e:  # comments disabled / quota / not found
            print(f"    [skip {video_id}] {e}")
            break
        for it in data.get("items", []):
            snip = it["snippet"]["topLevelComment"]["snippet"]
            out.append(snip.get("textDisplay", ""))
        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(0.1)
    return out[:max_comments]


def collect(video_ids, max_per_video, api_key, keep_all) -> pd.DataFrame:
    rows = []
    for vid in video_ids:
        raw = fetch_comments(vid, max_per_video, api_key)
        kept = 0
        for c in raw:
            txt = _clean_for_storage(c)
            if len(txt.split()) < 3:
                continue
            if keep_all or looks_romanized_punjabi(txt):
                rows.append({"text": txt, "video_id": vid})
                kept += 1
        print(f"  {vid}: {kept} kept / {len(raw)} fetched")
    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["collected_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY"))
    ap.add_argument("--videos", help="comma-separated video ids")
    ap.add_argument("--query", help="search query to find videos")
    ap.add_argument("--max-videos", type=int, default=15)
    ap.add_argument("--max-per-video", type=int, default=300)
    ap.add_argument("--keep-all", action="store_true",
                    help="skip the Romanized-Punjabi filter (keep every comment)")
    ap.add_argument("--out", type=Path, default=C.RAW_DIR / "youtube_comments.csv")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("ERROR: set YOUTUBE_API_KEY env var or pass --api-key")
    if not args.videos and not args.query:
        sys.exit("ERROR: provide --videos or --query")

    if args.videos:
        video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
    else:
        video_ids = search_video_ids(args.query, args.max_videos, args.api_key)

    print(f"collecting from {len(video_ids)} videos ...")
    df = collect(video_ids, args.max_per_video, args.api_key, args.keep_all)

    if df.empty:
        print("no comments collected — check quota / video ids / comments enabled.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nsaved {len(df)} unique comments -> {args.out}")
    print("next: python src/make_annotation_sheet.py --input", args.out)


if __name__ == "__main__":
    main()
