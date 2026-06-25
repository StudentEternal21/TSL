"""
Cebuano PinoyDictionary Sentence Scraper
========================================
Scrapes example sentences from cebuano.pinoydictionary.com and saves
them to a JSONL file with 'id' and 'text' columns only.

Install:
    pip install requests beautifulsoup4

Usage:
    # Scrape all letters:
    python scraper.py --output cebuano_sentences.jsonl

    # Scrape specific letters:
    python scraper.py --letters a b c --output cebuano_sentences.jsonl
"""

import argparse
import json
import logging
import string
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_URL    = "https://cebuano.pinoydictionary.com"
LIST_URL    = BASE_URL + "/list/{letter}/"
PAGE_URL    = BASE_URL + "/list/{letter}/{page}/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL + "/",
}

REQUEST_DELAY = 1.5   # seconds between page fetches
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3

ALL_LETTERS = list(string.ascii_lowercase)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── HTTP ───────────────────────────────────────────────────────────────────────

def fetch(url: str) -> str | None:
    """Fetch a URL with retry logic. Returns HTML string or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            wait = attempt * 2
            if attempt < MAX_RETRIES:
                log.warning(f"Attempt {attempt} failed ({exc}). Retrying in {wait}s…")
                time.sleep(wait)
            else:
                log.error(f"Giving up on {url}: {exc}")
                return None


# ── Parsing ────────────────────────────────────────────────────────────────────

def get_last_page(soup: BeautifulSoup) -> int:
    """Extract the total number of pages for a letter from the pagination block."""
    last_link = soup.find("a", title="Last Page")
    if last_link and last_link.get("href"):
        parts = last_link["href"].rstrip("/").split("/")
        try:
            return int(parts[-1])
        except ValueError:
            pass

    pages_block = soup.find("p", class_="pages")
    if pages_block:
        page_nums = []
        for a in pages_block.find_all("a"):
            try:
                page_nums.append(int(a.get_text(strip=True)))
            except ValueError:
                pass
        if page_nums:
            return max(page_nums)

    return 1


def parse_sentences(html: str) -> list[str]:
    """
    Parse word groups and extract sentences wrapped in <i> tags
    inside the definition blocks.
    """
    soup = BeautifulSoup(html, "html.parser")
    sentences = []

    for group in soup.find_all("div", class_="word-group"):
        def_div = group.find("div", class_="definition")
        if not def_div:
            continue
            
        # Target the requested <i> elements containing example sentences
        for i_tag in def_div.find_all("i"):
            sentence_text = i_tag.get_text(strip=True)
            # Normalize whitespace layout
            sentence_text = " ".join(sentence_text.split())
            
            # Simple threshold check to filter out empty tags or single punctuation markers
            if sentence_text and len(sentence_text) > 3:
                sentences.append(sentence_text)

    return sentences


# ── Scraping Orchestration ─────────────────────────────────────────────────────

def scrape_letter(letter: str, output_file, id_counter: int) -> tuple[int, int]:
    """Scrape all pages for one letter. Returns (sentences_written, updated_id_counter)."""
    page1_url = LIST_URL.format(letter=letter)
    log.info(f"[{letter.upper()}] Fetching page 1 → {page1_url}")
    html = fetch(page1_url)
    if not html:
        log.error(f"[{letter.upper()}] Could not load page 1, skipping letter.")
        return 0, id_counter

    soup = BeautifulSoup(html, "html.parser")
    total_pages = get_last_page(soup)
    log.info(f"[{letter.upper()}] {total_pages} page(s) found.")

    total_written = 0

    for page_num in range(1, total_pages + 1):
        page_url = (
            LIST_URL.format(letter=letter)
            if page_num == 1
            else PAGE_URL.format(letter=letter, page=page_num)
        )

        if page_num > 1:
            log.info(f"[{letter.upper()}] Fetching page {page_num}/{total_pages}…")
            html = fetch(page_url)
            if not html:
                log.error(f"[{letter.upper()}] Failed to load page {page_num}, skipping.")
                time.sleep(REQUEST_DELAY)
                continue

        sentences = parse_sentences(html)
        for sentence in sentences:
            line_data = {
                "id": f"ceb_{id_counter:02d}",
                "text": sentence
            }
            output_file.write(json.dumps(line_data, ensure_ascii=False) + "\n")
            id_counter += 1
            total_written += 1
            
        output_file.flush()
        log.info(f"[{letter.upper()}] Page {page_num}/{total_pages} → Extracted {len(sentences)} sentences")

        if page_num < total_pages:
            time.sleep(REQUEST_DELAY)

    return total_written, id_counter


# ── Counter Helper ─────────────────────────────────────────────────────────────

def get_starting_id(output_path: Path) -> int:
    """Finds the next logical ID index by checking the existing output file layout."""
    next_index = 1
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        parts = obj["id"].split("_")
                        if len(parts) == 2 and parts[1].isdigit():
                            next_index = max(next_index, int(parts[1]) + 1)
                except json.JSONDecodeError:
                    pass
    return next_index


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    global REQUEST_DELAY
    parser = argparse.ArgumentParser(
        description="Scrape sentences from cebuano.pinoydictionary.com to a JSONL corpus."
    )
    parser.add_argument(
        "--letters", "-l", nargs="+", default=None,
        metavar="LETTER",
        help="Letters to scrape (e.g. --letters a b c). Default: all a–z."
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output JSONL file path."
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY,
        help=f"Seconds between page requests (default: {REQUEST_DELAY})."
    )
    args = parser.parse_args()

    REQUEST_DELAY = args.delay
    letters = [l.lower() for l in args.letters] if args.letters else ALL_LETTERS

    # Automatically set the counter index depending on existing data state
    id_counter = get_starting_id(args.output)
    if id_counter > 1:
        log.info(f"Resuming pipeline append sequence. Starting ID: ceb_{id_counter:02d}")

    grand_total = 0
    with args.output.open("a", encoding="utf-8") as out_f:
        for letter in letters:
            count, id_counter = scrape_letter(letter, out_f, id_counter)
            grand_total += count
            log.info(f"[{letter.upper()}] Done. {count} sentences appended.")

    log.info(f"\nFinished. New sentences appended during this session: {grand_total}")


if __name__ == "__main__":
    main()