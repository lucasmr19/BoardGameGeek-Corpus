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
CORPORA_STATISTICS_DIR = f"{CORPORA_DIR}/statistics"
BALANCE_REPORTS_DIR = f"{PROCESSED_DIR}/balance_reports"
VECTORS_DIR = f"{PROCESSED_DIR}/vectors"
DATASETS_DIR = f"{PROCESSED_DIR}/datasets"
MODELS_DIR = f"{PROCESSED_DIR}/models"
LEXICONS_DIR = f"{DATA_DIR}/lexicons"
RANKS_DF = pd.read_csv(f"{RAW_DIR}/boardgames_ranks.csv")
BGG_STATS_DF = pd.read_csv(f"{CRAWLER_DIR}/bgg_stats.csv")
CORPUS_NAME = "bgg_corpus"