"""
Collect comments from r/apple, r/google, r/samsung via the Arctic Shift API,
for the full window 2024-01-01 to 2024-06-30. Writes data/{subreddit}_comments.jsonl
(one JSON object per line). No cap is applied: the whole window is paginated.

The collection is robust to transient API errors (retries with backoff) and
resumable: if a data file already exists, collection continues from the oldest
comment already saved instead of starting over.
"""

import json
import os
import time
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SUBREDDITS = ["apple", "google", "samsung"]
# Unix timestamps: 2024-01-01 00:00:00 UTC and 2024-07-01 00:00:00 UTC
AFTER_TS  = 1704067200
BEFORE_TS = 1719792000
LIMIT     = 100        # max per page allowed by Arctic Shift
SLEEP_SEC = 0.5        # delay between requests
OUT_DIR   = os.path.join(os.path.dirname(__file__), "data")
BASE_URL  = "https://arctic-shift.photon-reddit.com/api/comments/search"

os.makedirs(OUT_DIR, exist_ok=True)


def make_session():
    """A session that retries transient errors (429, 5xx) with backoff."""
    retry = Retry(
        total=8,
        backoff_factor=1.5,                       # 0, 1.5, 3, 6, ... seconds
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def fetch_page(session, subreddit, after, before):
    params = {"subreddit": subreddit, "after": after, "before": before, "limit": LIMIT}
    # extra manual retries if the session adapter still raises
    for attempt in range(5):
        try:
            resp = session.get(BASE_URL, params=params, timeout=45)
            if resp.status_code >= 500 or resp.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json().get("data", [])
        except requests.exceptions.RequestException:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    return []


def load_progress(out_path):
    """Resume support: return (seen ids, oldest timestamp) from an existing file."""
    seen, oldest = set(), None
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue           # skip a possible truncated last line
            cid, ts = c.get("id"), c.get("created_utc")
            if cid:
                seen.add(cid)
            if ts is not None and (oldest is None or ts < oldest):
                oldest = ts
    return seen, oldest


def collect(session, subreddit):
    """
    Page backwards in time from BEFORE_TS to AFTER_TS, writing every comment once.
    The API returns newest-first; the `before` cursor moves to the oldest timestamp
    of each page. Comments are de-duplicated by id and the cursor is forced one
    second back whenever a page yields nothing new, so dense seconds cannot stall
    the loop. If the output file already exists, collection resumes from there.
    """
    out_path = os.path.join(OUT_DIR, f"{subreddit}_comments.jsonl")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        seen, oldest = load_progress(out_path)
        cursor = oldest if oldest is not None else BEFORE_TS
        mode = "a"
        print(f"[{subreddit}] resuming: {len(seen):,} comments already saved, "
              f"continuing before {datetime.fromtimestamp(cursor, tz=timezone.utc):%Y-%m-%d}")
    else:
        seen, cursor, mode = set(), BEFORE_TS, "w"
        print(f"[{subreddit}] starting collection")

    page = 0
    with open(out_path, mode, encoding="utf-8") as fh:
        while cursor > AFTER_TS:
            batch = fetch_page(session, subreddit, AFTER_TS, cursor)
            if not batch:
                break

            new = [c for c in batch if c.get("id") not in seen]
            for c in new:
                seen.add(c["id"])
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            fh.flush()             # keep progress on disk for resuming

            oldest = min(c["created_utc"] for c in batch)
            page += 1
            when = datetime.fromtimestamp(oldest, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  page {page:4d} | +{len(new):3d} new | total {len(seen):,} | oldest {when}")

            if oldest <= AFTER_TS:
                break
            cursor = oldest if new and oldest < cursor else cursor - 1
            time.sleep(SLEEP_SEC)

    print(f"[{subreddit}] done - {len(seen):,} comments -> {out_path}\n")
    return len(seen)


if __name__ == "__main__":
    session = make_session()
    totals = {sub: collect(session, sub) for sub in SUBREDDITS}
    print("=== collection summary ===")
    for sub, n in totals.items():
        print(f"  r/{sub:<10}: {n:>8,} comments")
