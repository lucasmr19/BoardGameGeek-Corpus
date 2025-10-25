"""
Configuration constants for BoardGeekGames Corpus project.
"""
import pandas as pd

DATA_DIR = "data"
API_DIR = f"{DATA_DIR}/api"
RAW_DIR = f"{DATA_DIR}/raw"
CRAWLER_DIR = f"{DATA_DIR}/crawler"
PROCESSED_DIR = f"{DATA_DIR}/processed"
CORPORA_DIR = f"{PROCESSED_DIR}/corpora"
BALANCE_REPORTS_DIR = f"{PROCESSED_DIR}/balance_reports"
VECTORS_DIR = f"{PROCESSED_DIR}/vectors"
LEXICONS_DIR = f"{DATA_DIR}/lexicons"
RANKS_DF = pd.read_csv(f"{RAW_DIR}/boardgames_ranks.csv")
BGG_STATS_DF = pd.read_csv(f"{CRAWLER_DIR}/bgg_stats.csv")