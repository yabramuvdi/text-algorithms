from striprtf.striprtf import rtf_to_text
import os
from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq
import re


ARTICLES_PATH = "../../../EPU_data/raw_articles/MidCentury/labeled_articles/"
OUTPUT_PATH = "../../../EPU_data/extracted_articles/epu_midcentury_text.parquet"
N_ARTICLES_EXPECTED = 8757


extracted_text = {"article_num": [],
                  "text": []}

# List of filenames in the folder
filenames = os.listdir(ARTICLES_PATH)

logger.info(f"found {len(filenames)} files in the articles folder")

# List to store the article numbers where no text is extracted
articles_blank = []

for i in range(1, N_ARTICLES_EXPECTED + 1):
    # Each article file is a number, starting from 1

    try:
        logger.info(f"reading article {i} / {N_ARTICLES_EXPECTED}...")
        filename = f"{i}.rtf"
        with open(ARTICLES_PATH + filename, "r") as file:
            contents = file.read()
    except FileNotFoundError:
        logger.info(f"couldn't find article {i}")
        continue

    # Successfully read in the file, can now remove it from the list
    filenames.remove(filename)

    # # Get the text
    # text = rtf_to_text(contents)
    #
    # if text.strip() == "":
    #     print(i)
    #     # First method failed - now try splitting up by paragraph
    #     text = ""
    #     paras = contents.split("\\par")
    #     for para in paras:
    #         text += rtf_to_text(para)

    pattern = r"{\\cs[0-9]*\\lang1033\\langfe1033\\b[0-9]*\\i[0-9]*\\ul[0-9]*\\strike[0-9]*\\scaps[0-9]*\\fs[0-9]*\\afs[0-9]*\\charscalex[0-9]*\\expndtw-*[0-9]*\\cf[0-9]*\\dn[0-9]*.*?}"
    text = ""
    matches = re.finditer(pattern, contents, re.DOTALL)

    for match in matches:
        para = rtf_to_text(contents[match.start(0):match.end(0)])
        if para:
            text += "\n" + para

    if text.strip() == "":
        articles_blank.append(i)

    extracted_text["article_num"].append(i)
    extracted_text["text"].append(text)

if filenames:
    # Some filenames remaining in the list - weren't all read in
    logger.info("following files in the folder weren't read in:")
    print(*filenames, sep="\n")
else:
    logger.info("read in all files")

if articles_blank:
    logger.info(f"following {len(articles_blank)} articles extracted no text: ", end="")
    print(*articles_blank, sep=", ")
else:
    logger.info(f"all articles extracted")

extracted_text_tbl = pa.Table.from_pydict(extracted_text)

# Write to disk
logger.info("writing to disk...")
pq.write_table(extracted_text_tbl, OUTPUT_PATH)
