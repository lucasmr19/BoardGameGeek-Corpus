#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import os
import pandas as pd
import xml.etree.ElementTree as ET
import json
import time
import argparse

from ..resources import LOGGER

# ----------------------------
# Configuration
# ----------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "api") # BoardGeekGames-Corpus/data/api
IDS = [i for i in range(1,510)]
RETRY_WAIT = 2
RETRIES = 5

# ----------------------------
# Auxiliary functions
# ----------------------------
def fetch_with_retry(url, retries=RETRIES, wait=RETRY_WAIT):
    """Makes a request to the BGG API with retries if it returns 202 (processing)."""
    for attempt in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            return response
        elif response.status_code == 202:
            LOGGER.info(f"Retrying URL (processing) {url}... attempt {attempt + 1}")
            time.sleep(wait)
        else:
            response.raise_for_status()
    raise Exception(f"Could not get valid response from API for {url}.")

def parse_text(element):
    """Gets the text from an XML element, returns 'N/A' if it doesn't exist."""
    return element.text if element is not None else "N/A"

# ----------------------------
# Extraction functions
# ----------------------------

def get_text(game_xml, tag):
    """Gets the text from an XML field, returns 'N/A' if it doesn't exist."""
    element = game_xml.find(tag)
    return element.text if element is not None else "N/A"

def get_name(game_xml):
    """Returns the primary name of the game."""
    name_elem = game_xml.find("name[@primary='true']")
    return name_elem.text if name_elem is not None else "N/A"

def extract_metadata(game_xml, game_id):
    """Extracts metadata from a game from the XML (without comments)."""
    
    # Helper function to get text
    def get_text(element, tag):
        el = element.find(tag)
        return el.text if el is not None else "N/A"

    # Extract classifications in a subdictionary
    classifications = {
        "mechanics": [elem.text for elem in game_xml.findall("boardgamemechanic")],
        "categories": [elem.text for elem in game_xml.findall("boardgamecategory")],
        "families": [elem.text for elem in game_xml.findall("boardgamefamily")]
    }

    # Build main metadata dictionary
    metadata = {
        "id": game_id,
        "name": get_name(game_xml),
        "yearpublished": get_text(game_xml, "yearpublished"),
        "minplayers": get_text(game_xml, "minplayers"),
        "maxplayers": get_text(game_xml, "maxplayers"),
        "minplaytime": get_text(game_xml, "minplaytime"),
        "maxplaytime": get_text(game_xml, "maxplaytime"),
        "age": get_text(game_xml, "age"),
        "description": get_text(game_xml, "description"),
        "image": get_text(game_xml, "image"),
        "thumbnail": get_text(game_xml, "thumbnail"),
        "publishers": [p.text for p in game_xml.findall("boardgamepublisher")],
        "designers": [d.text for d in game_xml.findall("boardgamedesigner")],
        "artists": [a.text for a in game_xml.findall("boardgameartist")],
        "classifications": classifications
    }

    return metadata

def save_metadata(metadata, game_id, output_dir):
    """Saves game metadata to JSON."""
    output_file = f"{output_dir}/bgg_metadata_{game_id}_api.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    LOGGER.info(f"Metadata saved to {output_file}")

def save_reviews(comments, game_id, output_dir):
    """Saves game reviews to JSON."""
    output_file = f"{output_dir}/bgg_reviews_{game_id}_api.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=4)
    LOGGER.info(f"{len(comments)} comments saved to {output_file}")

def extract_comments(game_id, max_pages=None):
    """Downloads all reviews for a game (with comments included) with pagination."""
    comments = []
    page = 1

    while True:
        if max_pages is not None and page > max_pages:
            LOGGER.info(f"Limit of {max_pages} pages reached for game {game_id}.")
            break

        url = f"https://boardgamegeek.com/xmlapi/boardgame/{game_id}?comments=1&page={page}"
        response = fetch_with_retry(url)
        root = ET.fromstring(response.content)
        game_xml = root.find('boardgame')
        page_comments = game_xml.findall('comment')

        if not page_comments:
            LOGGER.info(f"No more comments available on page {page} for game {game_id}.")
            break

        for comment in page_comments:
            raw_rating = comment.attrib.get('rating', 'N/A')
            try:
                rating = float(raw_rating)
            except (TypeError, ValueError):
                rating = None
            comments.append({
                "username": comment.attrib.get('username', 'N/A'),
                "rating": rating,
                "comment": comment.text.strip() if comment.text else ''
            })

        LOGGER.info(f"Game {game_id}: page {page} downloaded with {len(page_comments)} comments.")
        page += 1

    return comments

def save_to_json(data, game_id, output_dir):
    """Saves data to a JSON file."""
    output_file = f"{output_dir}/bgg_reviews_{game_id}_api.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    LOGGER.info(f"Saved {len(data['comments'])} comments to {output_file}")

# ----------------------------
# Main flow
# ----------------------------
def process_game(game_id, output_dir, max_pages=None):
    """Processes a game: downloads metadata and comments separately."""
    LOGGER.info(f"Processing game {game_id}...")

    # --- Metadata ---
    url_game = f"https://boardgamegeek.com/xmlapi/boardgame/{game_id}"
    response = fetch_with_retry(url_game)
    root = ET.fromstring(response.content)
    game_xml = root.find("boardgame")
    metadata = extract_metadata(game_xml, game_id)
    save_metadata(metadata, game_id, output_dir)

    # --- Comments ---
    comments = extract_comments(game_id, max_pages=max_pages)
    save_reviews(comments, game_id, output_dir)

# ----------------------------
# Independent download
# ----------------------------
def process_metadata_only(game_id, output_dir):
    """Downloads and saves only game metadata."""
    LOGGER.info(f"Processing metadata for game {game_id}...")
    url_game = f"https://boardgamegeek.com/xmlapi/boardgame/{game_id}"
    response = fetch_with_retry(url_game)
    root = ET.fromstring(response.content)
    game_xml = root.find("boardgame")
    metadata = extract_metadata(game_xml, game_id)
    save_metadata(metadata, game_id, output_dir)


def process_reviews_only(game_id, output_dir, max_pages=None):
    """Downloads and saves only game reviews."""
    LOGGER.info(f"Processing reviews for game {game_id}...")
    comments = extract_comments(game_id, max_pages=max_pages)
    save_reviews(comments, game_id, output_dir)


# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BGG API Scraper - Downloads metadata and reviews from BoardGameGeek API"
    )
    parser.add_argument("--ids", type=int, nargs="+", default=IDS, help="List of game IDs")
    parser.add_argument(
        "--ranks-file",
        type=str,
        default="boardgames_ranks.csv",
        help="CSV file with game rankings (default: boardgames_ranks.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for data (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of comment pages to download per game (default: all pages)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "metadata", "reviews"],
        default="all",
        help="Processing mode: 'all' (metadata + reviews), 'metadata' only, or 'reviews' only (default: all)"
    )
    
    args = parser.parse_args()
    
    # Determine game IDs to process
    if args.ids:
        game_ids = args.ids
    else:
        # Try to load from CSV file
        if not os.path.exists(args.ranks_file):
            LOGGER.error(f"Ranks file not found: {args.ranks_file}")
            LOGGER.error("Please specify game IDs with --ids or provide a valid --ranks-file")
            exit(1)
        
        ranks_df = pd.read_csv(args.ranks_file)
        # Default: process first game in CSV if no IDs specified
        game_ids = ranks_df["id"].head(1).tolist()
        LOGGER.info(f"No IDs specified, using first game from {args.ranks_file}: {game_ids}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each game based on mode
    for game_id in game_ids:
        if args.mode == "all":
            process_game(game_id, args.output_dir, args.max_pages)
        elif args.mode == "metadata":
            process_metadata_only(game_id, args.output_dir)
        elif args.mode == "reviews":
            process_reviews_only(game_id, args.output_dir, args.max_pages)