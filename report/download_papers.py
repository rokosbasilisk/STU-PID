#!/usr/bin/env python3
"""
Bulk-download PDFs by title from arXiv with retry logic to handle intermittent connection issues.
Dependencies: pip install pandas arxiv tqdm python-slugify
"""

import argparse
import pathlib
import time
import arxiv
import pandas as pd
from slugify import slugify
from tqdm import tqdm
from urllib3.exceptions import ProtocolError, SSLError
import requests

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries

def download_title(title: str, client: arxiv.Client, out_dir: pathlib.Path):
    def safe_search(query):
        for attempt in range(MAX_RETRIES):
            try:
                return client.results(arxiv.Search(query=query, max_results=1, sort_by=arxiv.SortCriterion.Relevance))
            except (ProtocolError, SSLError, requests.exceptions.RequestException):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    return iter(())
        return iter(())

    # Try exact-title search
    search_exact = f'ti:"{title}"'
    results = safe_search(search_exact)
    art = None
    try:
        art = next(results)
    except StopIteration:
        # Fallback to keyword search
        results = safe_search(title)
        try:
            art = next(results)
        except StopIteration:
            tqdm.write(f"❌  not found: {title}")
            return

    fname = f"{slugify(title)[:120]}.pdf"
    path = out_dir / fname
    if path.exists():
        return

    for attempt in range(MAX_RETRIES):
        try:
            art.download_pdf(dirpath=out_dir, filename=fname)
            return
        except (ProtocolError, SSLError, requests.exceptions.RequestException) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                tqdm.write(f"⚠️  failed download: {title}: {e}")
                return
        except Exception as e:
            tqdm.write(f"⚠️  error with {title}: {e}")
            return

def main():
    parser = argparse.ArgumentParser(description="Download papers from arXiv by title.")
    parser.add_argument("csv", help="CSV file with a 'title' column")
    parser.add_argument("-o", "--out", default="pdfs", help="output directory")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    titles = pd.read_csv(args.csv)["title"].dropna().unique()[850:]

    client = arxiv.Client(page_size=1, delay_seconds=0.5)
    for title in tqdm(titles, desc="Downloading"):
        download_title(title, client, out_dir)

if __name__ == "__main__":
    main()

