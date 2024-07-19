import xml.etree.ElementTree as ET
import os
from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq


def process_articles(articles_path: str, output_path: str, n_articles_expected: int):
    """Read in all articles, extract text and combine into a parquet file, written to disk"""

    extracted_text = {"article_num": [],
                      "text": []}

    # List of filenames in the folder
    filenames = os.listdir(articles_path)

    logger.info(f"found {len(filenames)} files in the articles folder")

    # List to store the article numbers where no text is extracted
    articles_blank = []

    for i in range(1, n_articles_expected + 1):
        # Each article file is a number, starting from 1

        try:
            logger.info(f"reading article {i} / {n_articles_expected}...")
            filename = f"{i}.xml"
            tree = ET.parse(articles_path + filename)
        except FileNotFoundError:
            logger.info(f"couldn't find article {i}")
            continue

        # Successfully read in the file, can now remove it from the list
        filenames.remove(filename)

        # Get the text
        text = ""
        root = tree.getroot()
        paras = root.findall("./txtdt/text/paragraph")

        for para in paras:
            cht_list = para.findall("./cht")
            if para.text:
                text += para.text
            if cht_list:
                for cht in cht_list:
                    if cht.text:
                        text += cht.text

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
    pq.write_table(extracted_text_tbl, output_path)


# Process the historical articles
logger.info("processing historical articles...")
process_articles("../../../EPU_data/raw_articles/Historical/labeled_articles/",
                 "../../../EPU_data/extracted_articles/epu_historical_text.parquet",
                 2349)

# Process the historical oversample articles
logger.info("processing historical oversample articles...")
process_articles("../../../EPU_data/raw_articles/Historical Oversample/labeled_articles/",
                 "../../../EPU_data/extracted_articles/epu_historical_oversample_text.parquet",
                 960)


# # TESTING
# filename = "1.xml"
#
# tree = ET.parse(ARTICLES_PATH + filename)
# root = tree.getroot()
#
# # print(*[elem.tag for elem in root.iter()])
#
# # Function to recursively print the tree structure
# def print_tree(node, indent=""):
#     if hasattr(node, 'tag') and node.tag is not None:
#         print(indent + node.tag)
#         for child in node:
#             print_tree(child, indent + "  ")
#
# # Print the tree structure starting from the root
# print_tree(root)
#
# paras = root.findall("./txtdt/text/paragraph")
#
# for para in paras:
#     cht_list = para.findall("./cht")
#     if para.text:
#         print(para.text)
#     if cht_list:
#         for cht in cht_list:
#             if cht.text:
#                 print(cht.text)
