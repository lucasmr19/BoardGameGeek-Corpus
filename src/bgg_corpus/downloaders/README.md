[⬅ Back to bgg_corpus README](../README.md)

# Downloaders Submodule

The `downloaders` submodule provides tools for **downloading board game data** from BoardGameGeek (BGG) using either a **web crawler** or the **official API**.

```

downloaders/
├── bgg_api.py       # Downloader using BGG XML API
├── bgg_crawler.py   # Web crawler using Selenium

```

## Overview of Methods

### 1. BGG Crawler (`bgg_crawler.py`)

- **Mechanism:** Uses **Selenium** to navigate BGG web pages and extract review data.

- **Capabilities:**

  - Full flexibility in filtering reviews:

    - `rated=0/1` → include/exclude unrated/rated reviews
    - `rating=num` → filter reviews by a specific rating between 1–10
    - `commented=0/1` → include/exclude reviews without comments

  - Can extract **timestamps** of reviews.
  - Supports **balanced downloads per game** (equal number of reviews per rating, double weight for neutral ratings 5–6) via `process_game_balanced`.

- **Trade-offs:**

  - Slower than the API due to browser rendering and page loading.
  - Requires a Chrome driver and proper Selenium setup.
  - Best suited when **fine-grained filtering** or **review-level timestamps** are needed.
  - Ideal for **balancing the dataset directly from the source**, avoiding natural imbalances (most reviews cluster around ratings 6–7, very few at extremes like 1 or 10).

- **`--save` Flag:**

  The crawler provides a `--save` argument to control what data is written to disk:

  | Option    | Description                                                                                 |
  | --------- | ------------------------------------------------------------------------------------------- |
  | `reviews` | Only saves the raw review data (`.json`) for the specified games.                           |
  | `stats`   | Only saves aggregated statistics for each game (e.g., `total_all`, `total_commented`, etc.) |
  | `both`    | Saves **both** raw reviews and statistics (default behavior).                               |

  **Example Usage:**

  ```bash
  # Save only reviews
  python bgg_crawler.py --ids 50 51 52 --save reviews

  # Save only stats
  python bgg_crawler.py --ids 50 51 52 --save stats

  # Save both reviews and stats (default)
  python bgg_crawler.py --ids 50 51 52 --save both
  ```

**Notes:**

- If you use `--mode balanced`, the crawler will save reviews according to the per-rating balance settings when `--save reviews` or `--save both` is selected.

- Stats saved when `--save stats` or `--save both` include:

  ```
  total_all, total_commented, total_rated, total_rated_and_commented,
  avgweight, numweights, poll_avg, poll_votes
  ```

- These outputs are **essential for Phase 1** of the corpus-building workflow.

- **Example Usage:**

  ```bash
  python bgg_crawler.py --ids 50 51 52 --mode all --headless
  ```

### 2. BGG API (`bgg_api.py`)

- **Mechanism:** Uses the **official BGG XML API** to fetch game metadata and reviews.

- **Capabilities:**

  - Efficient extraction of **metadata** (game properties, designers, categories, mechanics).
  - Fast retrieval for **massive review datasets**.
  - Handles retries for `202 Processing` responses.

- **Limitations:**

  - Less flexible filtering; cannot easily exclude reviews by `rated`/`commented`.
  - Timestamps and advanced review flags may be limited or require extra parsing.
  - Not ideal for **balanced downloads per rating** — subsampling would be required afterward.

- **Example Usage:**

  ```bash
  python bgg_api.py --ids 50 51 52 --mode all
  ```

---

## Key Differences

| Feature                   | Crawler                                            | API                          |
| ------------------------- | -------------------------------------------------- | ---------------------------- |
| Filtering options         | ✅ Flexible (`rated`, `commented`, rating, custom) | ❌ Limited                   |
| Timestamps extraction     | ✅ Can parse exact timestamps                      | ⚠ Partial or slower to parse |
| Metadata extraction       | ⚠ Slower, needs page parsing                       | ✅ Fast and structured       |
| Speed                     | ⚠ Slower                                           | ✅ Fast for large datasets   |
| Scale / Massive downloads | ⚠ Not optimal                                      | ✅ Better suited for bulk    |
| Balancing reviews         | ✅ Supports per-rating balanced downloads          | ❌ Needs post-processing     |
| Setup                     | Selenium + Chrome driver needed                    | Standard `requests` library  |

---

## Summary

- **Use the Crawler** when you need:

  - Fine-grained filtering of reviews.
  - Exact timestamps of user comments.
  - Balanced downloads per rating to handle natural review imbalances.
  - Controlled page-by-page scraping.

- **Use the API** when you need:

  - Rapid metadata extraction.
  - Large-scale review downloads.
  - Structured, reliable XML data with retries.

Both methods complement each other and can be combined depending on the **dataset size, filtering needs, and balancing requirements**.

## 🔍 Further Discussion: Recommendation to Build a Balanced Corpus with Maximum Information

To construct a **rich and fully balanced BGG corpus**, we recommend running both downloaders in a coordinated workflow:

1. **Crawler (`bgg_crawler.py`)**

   - Use the flags:

     ```bash
     --save both --mode balanced
     ```

   - This ensures that:

     - **All raw reviews** are saved to JSON files.
     - **Aggregated statistics** for each game are generated.
     - **Balanced review extraction per rating** is applied, so extreme ratings (1–2, 9–10) are proportionally represented alongside common ratings (5–6, 6–7).
     - Timestamps, comment flags, and other review-level details are retained.

   - **Purpose:** Provides the **core dataset for preprocessing and balancing**, allowing accurate oversampling/undersampling strategies later in Phase 1 of `build_corpus()`.

2. **API (`bgg_api.py`)**

   - Run with:

     ```bash
     --mode metadata
     ```

   - This ensures that:

     - Full **game-level metadata** is retrieved (categories, designers, mechanics, publication info).
     - Fast and structured extraction for all requested game IDs.
     - Metadata can later be merged with review-level data to enrich `GameCorpus` objects.

3. **Merging and Standardization**

   - Both outputs are **merged and standardized in Phase 1** of the `build_corpus()` workflow.
   - Ensures consistency of IDs, review formats, and metadata fields before any preprocessing, balancing, or text processing is applied.
   - Guarantees that the final corpus includes:

     - Balanced representation of review ratings.
     - Complete game metadata.
     - All essential stats for quality control and downstream analysis.

4. **Output Locations**

   Each downloader saves its `.json` files automatically:

   | Path                 | Contents                     |
   | :------------------- | :--------------------------- |
   | `../../data/crawler` | Raw review data + statistics |
   | `../../data/api`     | Metadata and/or API reviews  |

5. **Benefits of Using Both Downloaders Together**

   - The **crawler output** feeds detailed review-level information into the balancing phase.
   - The **API output** enriches each `GameCorpus` with descriptive fields for NLP, analytics, or machine learning tasks.
   - Using both ensures the **maximum information** is captured, producing a corpus that is:

     - **Balanced** across ratings.
     - **Rich in metadata**.
     - **Ready for preprocessing, feature extraction, and downstream tasks**.

> ⚠️ **Best Practice:** Always run the crawler first to obtain balanced reviews and stats, then run the API for metadata. Do not skip either step if your goal is a fully informative corpus.
