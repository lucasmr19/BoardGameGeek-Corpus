#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGG crawler downloader from BoardGameGeek web page.
Change IDS = [...] to test different games.
Change OUTPUT_DIR to save data to a different directory.
"""

import os
import json
import pandas as pd
import re
import requests
import argparse
from datetime import datetime, timedelta
from typing import Optional, Any
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

from resources import LOGGER

# ---------- CONFIG ----------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "crawler")
PAGE_SLEEP = 1
RETRY_WAIT = 2
RETRIES = 3
IDS = [i for i in range(50, 510)]  # Test with first 20 game IDs

# ---------- HELPERS ----------
def normalize_timestamp(ts: Optional[str]) -> Optional[int]:
    """Converts 'Last updated: ...' or 'Today/Yesterday at ...' to timestamp."""
    if not ts:
        return None
    ts = ts.strip().lower()
    months_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,
                  "aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12,
                  "ene":1,"abr":4,"ago":8,"dic":12}
    try:
        if ts.startswith("last updated:"):
            ts = ts.replace("last updated:", "").strip()
        if ts.startswith("today at"):
            h, m = map(int, ts.replace("today at","").strip().split(":"))
            now = datetime.now()
            return int(datetime(now.year, now.month, now.day, h, m).timestamp())
        if ts.startswith("yesterday at"):
            h, m = map(int, ts.replace("yesterday at","").strip().split(":"))
            now = datetime.now() - timedelta(days=1)
            return int(datetime(now.year, now.month, now.day, h, m).timestamp())
        parts = ts.split()
        if len(parts) == 3:
            day = int(parts[0])
            month = months_map.get(parts[1])
            year = int(parts[2])
            if month:
                return int(datetime(year, month, day).timestamp())
    except:
        return None
    return None

def init_driver(headless: bool = True) -> webdriver.Chrome:
    """Initializes Selenium Chrome with basic options."""
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
    return webdriver.Chrome(options=options)

def normalize_flag(flag: Any) -> Optional[int]:
    """Converts 1/0/True/False/'1'/'0'/None to 1/0/None."""
    if flag is None: return None
    if isinstance(flag, bool): return int(flag)
    if isinstance(flag, int) and flag in (0,1): return flag
    if isinstance(flag, str):
        s = flag.strip()
        if s in ("0","1"): return int(s)
    return None

def build_ratings_page_url(base_ratings_url: str, page: int = 1,
                           comment: Any = None, rated: Any = None,
                           rating: Optional[int] = None) -> str:
    """Builds BGG ratings page URL with comment/rated/rating flags."""
    comment_n = normalize_flag(comment)
    rated_n = normalize_flag(rated)
    base = base_ratings_url.split("?")[0].rstrip("/")
    url = f"{base}?pageid={page}"
    if comment_n is not None: url += f"&comment={comment_n}"
    if rated_n is not None: url += f"&rated={rated_n}"
    if rating is not None and 1 <= rating <= 10: url += f"&rating={rating}"
    return url

def get_total_reviews_count(base_ratings_url: str, driver: webdriver.Chrome,
                            comment: Any = None, rated: Any = None,
                            rating: Optional[int] = None,
                            page: int = 1, timeout: int = 10) -> Optional[int]:
    """
    Gets the total number of reviews from a BGG ratings page.
    Handles Angular rendering issues (placeholders like 999,999) by waiting and using regex.
    Returns:
        int: total number of reviews, 0 if none, or None if extraction fails.
    """
    url = build_ratings_page_url(base_ratings_url, page=page, comment=comment, rated=rated, rating=rating)
    LOGGER.info(f"[get_total_reviews_count] Checking total at: {url}")

    try:
        driver.get(url)

        # Wait until 'No items found' or the total count appears
        WebDriverWait(driver, timeout).until(
            lambda d: (
                "No items found" in d.page_source or
                re.search(r'of\s*<strong class="ng-binding">[\d,]+</strong>', d.page_source)
            )
        )

        html = driver.page_source

        # Case 1: 'No items found'
        if "No items found" in html:
            LOGGER.info("[get_total_reviews_count] Page indicates 'No items found'. Total = 0")
            return 0

        # Case 2: extract total using regex
        match = re.search(r'of\s*<strong class="ng-binding">([\d,]+)</strong>', html)
        if match:
            total_str = match.group(1).replace(",", "")
            total = int(total_str)
            # ignore obvious placeholders
            if total >= 999_000:
                LOGGER.warning(f"[get_total_reviews_count] Placeholder detected ({total}), retrying...")
                time.sleep(2)  # wait a bit and try again
                return get_total_reviews_count(base_ratings_url, driver, comment, rated, rating, page, timeout)
            LOGGER.info(f"[get_total_reviews_count] Total detected: {total}")
            return total

        # Fallback: save HTML for debugging
        html_path = os.path.join(OUTPUT_DIR, "debug_total_page.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        LOGGER.warning(f"[get_total_reviews_count] Could not detect total. HTML saved at {html_path}")

    except Exception as e:
        LOGGER.error(f"[get_total_reviews_count] Error extracting total: {e}")

    return None

def get_weight_stats(game_id: int) -> dict:
    """Extracts weight/complexity metrics from geekitemPreload on BGG."""
    url = f"https://boardgamegeek.com/boardgame/{game_id}"
    try:
        html = requests.get(url, timeout=10).text
        match = re.search(r"GEEK\.geekitemPreload\s*=\s*(\{.*?\});", html, re.DOTALL)
        if not match:
            return {}
        data = json.loads(match.group(1))

        stats = data.get("item", {}).get("stats", {})
        polls = data.get("item", {}).get("polls", {}).get("boardgameweight", {})

        return {
            "avgweight": float(stats.get("avgweight", 0)),
            "numweights": int(stats.get("numweights", 0)),
            "poll_avg": float(polls.get("averageweight", 0)),
            "poll_votes": int(polls.get("votes", 0)),
        }
    except Exception as e:
        LOGGER.warning(f"Could not extract weight stats from {url}: {e}")
        return {}

def save_bgg_stats(stats: dict, output_dir: str, csv_name: str = "bgg_stats.csv"):
    out_csv = os.path.join(output_dir, csv_name)
    if os.path.exists(out_csv):
        existing_df = pd.read_csv(out_csv)
    else:
        existing_df = pd.DataFrame()
    new_df = pd.DataFrame([stats])
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset=["game_id"], keep="last", inplace=True)
    combined_df.to_csv(out_csv, index=False)
    LOGGER.info(f"Saved statistics to {out_csv}")


def download_reviews(base_ratings_url: str, driver: webdriver.Chrome,
                     comment: Any = None, rated: Any = None,
                     rating: Optional[int] = None,
                     max_pages: Optional[int] = None,
                     limit: Optional[int] = None):
    """
    Downloads reviews for a given game, rating and comment/rated filters.
    If `limit` is set, stops when that many reviews have been collected.
    """
    comment_n = normalize_flag(comment)
    rated_n = normalize_flag(rated)
    reviews = []
    seen = set()
    page = 1
    
    # Initialize limit
    if limit is None:
        limit = float("inf")


    while True:
        if max_pages is not None and page > max_pages:
            break
        if len(reviews) >= limit:
            break  # Stop early if limit reached

        url = build_ratings_page_url(base_ratings_url, page=page,
                                     comment=comment_n, rated=rated_n,
                                     rating=rating)
        loaded = False
        LOGGER.info(f"Downloading review page {page}: {url}")

        for _ in range(RETRIES):
            try:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.summary-rating-item"))
                )
                loaded = True
                break
            except Exception:
                time.sleep(RETRY_WAIT)
        if not loaded:
            break

        items = driver.find_elements(By.CSS_SELECTOR, "li.summary-rating-item")
        if not items:
            break
        
        new_added = 0
        for item in items:
            if limit is not None and len(reviews) >= limit:
                break  # Stop mid-page if limit reached

            try:
                username_elem = item.find_element(By.CSS_SELECTOR, "a.comment-header-user")
                username = username_elem.get_attribute("innerText").strip()
            except:
                username = None
            try:
                rating_elem = item.find_element(By.CSS_SELECTOR, "div.rating-angular")
                rating_value = rating_elem.get_attribute("innerText").strip()
                rating_value = float(rating_value) if rating_value else None
            except:
                rating_value = None
            try:
                comment_elem = item.find_element(By.CSS_SELECTOR, "div.comment-body p")
                comment_text = comment_elem.get_attribute("innerText").strip()
            except:
                comment_text = None
            try:
                ts_elem = item.find_element(By.CSS_SELECTOR, "span.ng-binding")
                ts_title = ts_elem.get_attribute("title") or ts_elem.text
                timestamp = normalize_timestamp(ts_title)
            except:
                timestamp = None

            key = (username, rating_value, timestamp)
            if key not in seen:
                seen.add(key)
                reviews.append({
                    "username": username,
                    "rating": rating_value,
                    "comment": comment_text,
                    "timestamp": timestamp
                })
                new_added += 1

        if len(reviews) >= limit or new_added == 0:
            break  # Ensure no more pages fetched than needed

        page += 1
        time.sleep(PAGE_SLEEP)

    return reviews


def process_game_balanced(game_id: int, driver: webdriver.Chrome, output_dir: str, max_pages: Optional[int] = None):
    """
    Balanced download: equal number of reviews per rating (1–10),
    with double reviews for neutral ratings (5–6).
    Skip the game entirely if any rating has 0 reviews.
    """
    LOGGER.info(f"Processing game {game_id} in BALANCED mode ...")
    game_url = f"https://boardgamegeek.com/boardgame/{game_id}"
    driver.get(game_url)
    try:
        canonical = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "link[rel='canonical']"))
        )
        game_url = canonical.get_attribute("href")
    except:
        pass

    ratings_url = f"{game_url}/ratings"

    # --- Step 1: Count reviews per rating ---
    counts_per_rating = {}
    for rating in [1,2,10]: # Consider only extreme ratings cause they tend to be less
        total = get_total_reviews_count(ratings_url, driver, comment=1, rating=rating)
        total = total or 0
        counts_per_rating[rating] = total

    # If any rating has 0 reviews, skip the game
    if any(count == 0 for count in counts_per_rating.values()):
        LOGGER.warning(f"Game {game_id} has 0 reviews for at least one rating. Skipping balanced download.")
        return

    min_count = min(counts_per_rating.values())
    LOGGER.info(f"Using base limit per rating: {min_count} (neutral ratings will use double)")

    # --- Step 2: Download reviews ---
    balanced_reviews = []
    for rating in range(1, 11):
        limit = min_count * 2 if rating in [5,6] else min_count
        LOGGER.info(f"Downloading rating={rating} (limit={limit})")
        reviews = download_reviews(ratings_url, driver, comment=1, rating=rating, limit=limit)
        balanced_reviews.extend(reviews)

    # --- Step 3: Save results ---
    out_json = os.path.join(output_dir, f"bgg_reviews_{game_id}_crawler.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(balanced_reviews, fh, ensure_ascii=False, indent=2)

    LOGGER.info(f"Saved {len(balanced_reviews)} balanced reviews to {out_json}")

def process_game(game_id: int, driver: webdriver.Chrome, output_dir: str, max_pages: Optional[int] = None):
    """Processes a single game ID: downloads reviews and collects statistics."""
    LOGGER.info(f"Processing game {game_id} ...")
    game_url = f"https://boardgamegeek.com/boardgame/{game_id}"
    driver.get(game_url)
    try:
        canonical = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "link[rel='canonical']"))
        )
        game_url = canonical.get_attribute("href")
    except: 
        pass
    ratings_url = f"{game_url}/ratings"

    # --- Download reviews ---
    LOGGER.info("Downloading reviews with comments and ratings...")
    data = download_reviews(ratings_url, driver, comment=1, rated=1, max_pages=max_pages)

    # --- Save reviews to JSON ---
    out_json = os.path.join(output_dir, f"bgg_reviews_{game_id}_crawler.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    LOGGER.info(f"Saved {len(data)} reviews to {out_json}")

# ---------- MAIN ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BGG Crawler - Downloads reviews and statistics")
    parser.add_argument("--ids", type=int, nargs="+", default=IDS, help="List of game IDs")
    parser.add_argument("--mode", type=str, choices=["all", "balanced"], default="all", help="Download mode")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--stats-csv", type=str, default="bgg_stats.csv", help="CSV file for statistics")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to download")
    parser.add_argument("--headless", action="store_true", default=True, help="Headless mode")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="Show browser window")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    driver = init_driver(headless=args.headless)

    try:
        for game_id in args.ids:
            #if args.mode == "balanced":
            #    process_game_balanced(game_id, driver, args.output_dir, args.max_pages)
            #else:
            #    process_game(game_id, driver, args.output_dir, args.max_pages)

            # Save CSV statistics after processing
            game_url = f"https://boardgamegeek.com/boardgame/{game_id}"
            driver.get(game_url)
            try:
                canonical = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "link[rel='canonical']"))
                )
                game_url = canonical.get_attribute("href")
            except: 
                pass
               
            ratings_url = f"{game_url}/ratings"

            total_all = get_total_reviews_count(ratings_url, driver, comment=None, rated=None)
            total_commented = get_total_reviews_count(ratings_url, driver, comment=1, rated=None)
            total_rated = get_total_reviews_count(ratings_url, driver, comment=None, rated=1)
            total_rated_and_commented = get_total_reviews_count(ratings_url, driver, comment=1, rated=1)

            LOGGER.info(f"Totals -> all: {total_all}, commented: {total_commented}, "
                        f"rated: {total_rated}, rated+commented: {total_rated_and_commented}")

            stats = {
                "game_id": game_id,
                "total_all": total_all,
                "total_commented": total_commented,
                "total_rated": total_rated,
                "total_rated_and_commented": total_rated_and_commented,
            }

            stats.update(get_weight_stats(game_id))
            save_bgg_stats(stats, args.output_dir, args.stats_csv)
    finally:
        driver.quit()