"""
download_data.py — Fetch a handful of Gutenberg books for DCM training
======================================================================
Usage (Kaggle):
    !python /kaggle/working/dcm/download_data.py

Downloads ~20 books into /kaggle/working/gutenberg_texts/
"""

import csv
import os
import requests

CSV_PATH = "/kaggle/input/datasets/mateibejan/15000-gutenberg-books/gutenberg_metadata.csv"
OUT_DIR = "/kaggle/working/gutenberg_texts"
MAX_BOOKS = 20


def guess_id_column(row):
    for col in ("id", "Text#", "bookid", "book_id", "ID", "Nr."):
        if col in row:
            return col
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"CSV has {len(rows)} rows. Columns: {list(rows[0].keys())}")

    count = 0
    for row in rows:
        if count >= MAX_BOOKS:
            break
        link = row.get("Link", "")
        book_id = link.rstrip("/").split("/")[-1]
        if not book_id or not book_id.isdigit():
            continue
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and len(r.text) > 1000:
                path = os.path.join(OUT_DIR, f"{book_id}.txt")
                with open(path, "w", encoding="utf-8") as out:
                    out.write(r.text)
                count += 1
                print(f"  [{count}/{MAX_BOOKS}] Book {book_id}: {len(r.text):,} chars")
        except Exception as e:
            print(f"  Skipping {book_id}: {e}")

    print(f"\nDone — {count} books saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
