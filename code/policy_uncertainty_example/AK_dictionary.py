import re
import pyarrow as pa
import pyarrow.parquet as pq
# import wordsegment as ws
import string
from loguru import logger


# Initialise wordsegment library
# ws.load()


# Constants
DATA_PATH = "../../../EPU_data/extracted_articles/"
OUTPUT_PATH = "../../../text-algorithms-benchmarking/output/"

UNCERTAINTY_TERMS = ["uncertain", "uncertainty"]
ECONOMIC_TERMS = ["economic", "economy"]
POLICY_TERMS = ["regulation", "federal reserve", "deficit", "congress", "legislation", "white house"]

PUNCTUATION = string.punctuation.replace("-", "").replace("'", "")


def clean_text(text: str) -> str:
    """Remove punctuation, make lowercase. Returns blank string if provided with either blank string or None."""
    return text.lower().translate(str.maketrans('', '', PUNCTUATION)) if text else ""


# segment_and_clean_counter = 0
# segment_failure_count = 0
#
#
# def segment_and_clean(text: str) -> str:
#     """Use segmentation library to split text into most probable word divisions, after first using the standard
#     cleaning function"""
#
#     global segment_and_clean_counter, segment_failure_count
#     segment_and_clean_counter += 1
#     logger.info(f"segment + clean article {segment_and_clean_counter}")
#
#     cleaned = clean_text(text)
#
#     try:
#         return " ".join(ws.segment(cleaned))
#     except RecursionError:
#         logger.info("failed to segment")
#         segment_failure_count += 1
#         return cleaned


def label_articles(search_terms: list[str]) -> tuple[list[list[str]], list[int]]:
    """Search for the provided terms in the articles, return lists of the terms matched and the final label for each
     article"""

    terms_matched = []
    labels = []

    patterns = [r"\s+" + r"\s+".join(phrase.split()) + r"\s+" for phrase in search_terms]

    for text in all_articles_df["text"]:
        if not text:
            # Either None or blank string encountered - don't search
            terms_matched.append([])
            labels.append(0)
            continue

        matches = []
        for i, pattern in enumerate(patterns):
            if re.search(pattern, text):
                matches.append(search_terms[i])

        terms_matched.append(matches)
        labels.append(1 if matches else 0)

    return terms_matched, labels


# Read in the data
logger.info("reading in the articles...")

historical = pq.read_table(DATA_PATH + "epu_historical_text.parquet",
                           schema=pa.schema([pa.field("article_num", pa.string()),
                                             pa.field("text", pa.string())]))
historical_oversample = pq.read_table(DATA_PATH + "epu_historical_oversample_text.parquet",
                                      schema=pa.schema([pa.field("article_num", pa.string()),
                                                        pa.field("text", pa.string())]))
midcentury = pq.read_table(DATA_PATH + "epu_midcentury_text.parquet",
                           schema=pa.schema([pa.field("article_num", pa.string()),
                                             pa.field("text", pa.string())]))
modern = pq.read_table(DATA_PATH + "epu_modern_text.parquet")


# Concatenate tables
logger.info("concatenating the tables...")

historical = historical.rename_columns(["article", "text"])
historical = historical.append_column("time_period", pa.array(["historical"] * len(historical), pa.string()))

historical_oversample = historical_oversample.rename_columns(["article", "text"])
historical_oversample = historical_oversample.append_column("time_period",
                                                            pa.array(
                                                                ["historical_oversample"] * len(historical_oversample),
                                                                pa.string()))

midcentury = midcentury.rename_columns(["article", "text"])
midcentury = midcentury.append_column("time_period", pa.array(["midcentury"] * len(midcentury), pa.string()))

modern = modern.append_column("time_period", pa.array(["modern"] * len(modern), pa.string()))

all_articles = pa.concat_tables([historical, historical_oversample, midcentury, modern])

# Write combined articles table to disk
# logger.info("writing combined articles table to disk...")
# pq.write_table(all_articles, DATA_PATH + "epu_all_text.parquet")


# Clean the data
logger.info("cleaning the data...")
all_articles_df = all_articles.to_pandas()

# Take random subsample
# all_articles_df = all_articles_df.groupby("time_period").sample(n=10).reset_index()

logger.info(f"{len(all_articles_df)} articles")
all_articles_df["text"] = all_articles_df["text"].apply(clean_text)

# if segment_failure_count > 0:
#     logger.info(f"{segment_failure_count} articles weren't segmented due to recursion limit")


# Uncertainty
logger.info("uncertainty labels...")

uncertainty_matches, uncertainty_labels = label_articles(UNCERTAINTY_TERMS)
all_articles_df["uncertainty_matches"] = uncertainty_matches
all_articles_df["uncertainty_label"] = uncertainty_labels

# Economic
logger.info("economic labels...")

economic_matches, economic_labels = label_articles(ECONOMIC_TERMS)
all_articles_df["economic_matches"] = economic_matches
all_articles_df["economic_label"] = economic_labels

# Policy
logger.info("policy labels...")

policy_matches, policy_labels = label_articles(POLICY_TERMS)
all_articles_df["policy_matches"] = policy_matches
all_articles_df["policy_label"] = policy_labels


# Final EPU label (1 if an article has label 1 for all uncertainty, economic and policy)
logger.info("EPU labels...")

all_articles_df["epu_label"] = (all_articles_df["uncertainty_label"] * all_articles_df["economic_label"]
                                * all_articles_df["policy_label"])


logger.info("writing to disk...")
all_articles_df.drop(columns=["text"]).to_csv(OUTPUT_PATH + "AK_policy_uncertainty_dictionary.csv", index=False)
# all_articles_df.to_csv(OUTPUT_PATH + "AK_policy_uncertainty_dictionary_with_wordsegment.csv")
