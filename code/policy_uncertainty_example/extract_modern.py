""" Extracts the text from the HTML articles from the EPU data
"""

#%%

import pandas as pd
from bs4 import BeautifulSoup 


# main paths
LABELS_FILE = "../../data/epu_modern_labels.csv"
ARTICLES_PATH = "../../../EPU_data/raw_articles/Modern/labeled_articles/"
OUTPUT_PATH = "../../../EPU_data/extracted_articles/"


def get_text_paras(article_content) -> str:
    if article_content is None:
        return ""

    # Remove extra stuff amongst the article text e.g. caption at the end, images
    for div in article_content.find_all("div"):
        div.decompose()

    # Extract all remaining paragraphs of article text
    paragraphs = article_content.find_all('p') if article_content else []

    # Get the text of each paragraph within the body
    article_text = "".join(para.get_text() for para in paragraphs)

    return article_text


def get_text_sf(soup: BeautifulSoup) -> str:
    # Skip the paragraphs containing email address and info about where article appeared
    dtlcomment = soup.find_all("p", class_="dtlcomment")
    for p in dtlcomment:
        p.decompose()

    pageno = soup.find("p", id="pageno")
    if pageno:
        pageno.decompose()

    # Get article text from correct element
    article_content = (soup.body
                       .find("div", id="fontprefs_bottom"))
    if article_content:
        return get_text_paras(article_content)

    text1md = soup.body.find("td", class_="text1md")
    if text1md:
        article_content = text1md.font
        return get_text_paras(article_content)


def get_text_dmnb_mihb(soup: BeautifulSoup) -> str:

    content_id = (soup.body
                  .find("div", id="content"))

    if content_id:
        article_content = (content_id
                           .find("div", class_="docBody")
                           .find("div", class_="mainText"))

        caption = article_content.find("div")
        if caption:
            caption.decompose()

        return article_content.get_text()

    return ""


def get_text_la(soup: BeautifulSoup) -> str:

    article_content = (soup.body
                       .find("div", id="fullTextZoneId", class_="t-zone"))
    if article_content:
        return get_text_paras(article_content)

    article_content = (soup.body
                       .find("div", id="abstractZone"))

    return get_text_paras(article_content)


def get_text_nyt(soup: BeautifulSoup) -> str:

    article_content = (soup.body
                       .find_all("div", class_="mod-nytimesarticletext mod-articletext"))

    if article_content:
        return "".join(get_text_paras(content) for content in article_content)

    # First method didn't work

    article_id = (soup.body
                  .find("div", id="article"))
    if article_id:
        # Sometimes articleBody is the ID, sometimes the class
        article_content = (article_id
                           .find_all("div", id="articleBody"))
        if article_content:
            return "".join(get_text_paras(content) for content in article_content)

        article_content = (article_id
                           .find_all("div", class_="articleBody"))
        if article_content:
            return "".join(get_text_paras(content) for content in article_content)

        # Sometimes the file contains only the abstract/summary of the article
        article_content = (article_id
                           .find("p", class_="summaryText"))
        if article_content:
            return article_content.get_text()

    # Second method didn't work

    content_id = soup.body.find("div", id="content")
    if content_id:
        article_content = (content_id
                           .find("div", class_="entry-content"))
        # For some reason, a few articles have entry-content div inside another entry-content div
        inner_entry_content = (article_content
                               .find("div", class_="entry-content"))
        return get_text_paras(inner_entry_content if inner_entry_content else article_content)

    # Third method didn't work

    article_content = soup.body.find("nyt_text")
    if article_content:
        return get_text_paras(article_content)

    # Failed to extract text

    return ""


#%%

# load labels
df_complete = pd.read_csv(LABELS_FILE)

#%%

#===============
# Find articles
#===============

all_text = []
articles = df_complete["unique_id_current"].values

# with open("no_text_extracted.txt", "r") as file:
#     articles = file.read().split(", ")

unique_articles = pd.unique(articles)

if len(articles) != len(unique_articles):
    duplicates = articles.tolist()
    for article in unique_articles:
        duplicates.remove(article)
    print(f"{len(duplicates)} DUPLICATED ARTICLES:", end=" ")
    print(*duplicates, sep=", ")

n_articles_expected = len(unique_articles)

articles_not_found = []
articles_blank = []

print(f"expecting {n_articles_expected} articles")

for i, article in enumerate(unique_articles):

    file_path = ARTICLES_PATH + article + ".html"

    try:
        # Open the HTMl file and read its contents
        with open(file_path, "r", encoding="latin-1") as file:
            index = file.read()
        print(f"{article} ({i+1}/{n_articles_expected})")
    except FileNotFoundError:
        print(f"couldn't find {article}")
        articles_not_found.append(article)
        continue

    # Creating a BeautifulSoup object and specifying the parser
    soup = BeautifulSoup(index, 'lxml')

    # # Function to recursively print the tree structure
    # def print_tree(node, indent=""):
    #     if hasattr(node, 'name') and node.name is not None:
    #         print(indent + node.name)
    #         for child in node.children:
    #             print_tree(child, indent + "  ")

    # # Print the tree structure starting from the root
    # print_tree(Parse)

    if article[:2] == "SF":
        article_text = get_text_sf(soup)
    elif article[:2] == "LA":
        article_text = get_text_la(soup)
    elif article[:3] == "NYT":
        article_text = get_text_nyt(soup)
    elif article[:4] in ("DMNB", "MIHB"):
        article_text = get_text_dmnb_mihb(soup)
    else:
        print(f"{article} not a recognised publication")
        article_text = ""
        articles_not_found.append(article)

    if article_text.strip() == "":
        articles_blank.append(article)

    all_text.append(article_text)

if articles_not_found or articles_blank:
    print(f"extracted {n_articles_expected - len(articles_not_found) - len(articles_blank)}/{n_articles_expected}")
    print(f"following {len(articles_not_found)} articles not found: ", end="")
    print(*articles_not_found, sep=", ")
    print(f"following {len(articles_blank)} articles extracted no text: ", end="")
    print(*articles_blank, sep=", ")
else:
    print(f"all {n_articles_expected} articles extracted")

all_text_df = pd.DataFrame(data={"article": unique_articles,
                                 "text": all_text})
all_text_df.to_csv(OUTPUT_PATH + "epu_modern_text.csv")

# %%
