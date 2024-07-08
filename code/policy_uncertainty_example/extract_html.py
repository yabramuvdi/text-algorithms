""" Extracts the text from the HTML articles from the EPU data
"""

#%%

import pandas as pd
import numpy as np
import yaml
import os
from bs4 import BeautifulSoup 

# main paths
data_path = "../../data/"

# EDIT this path
articles_path = "../../../EPU/All Audit Hard Copies/All Audit Hard Copies/Modern/"

#%%

# load labels
df_complete = pd.read_csv(data_path + "epu_modern_labels.csv")

#%%

#===============
# Find articles
#===============

all_text = []
for article in df_complete["unique_id_current"].values:
    print(article)

    file_path = articles_path + article + ".html"
  
    # Opening the html file 
    HTMLFile = open(file_path, "r") 
  
    # Reading the file
    index = HTMLFile.read() 

    # Creating a BeautifulSoup object and specifying the parser 
    Parse = BeautifulSoup(index, 'lxml') 

    # # Function to recursively print the tree structure
    # def print_tree(node, indent=""):
    #     if hasattr(node, 'name') and node.name is not None:
    #         print(indent + node.name)
    #         for child in node.children:
    #             print_tree(child, indent + "  ")

    # # Print the tree structure starting from the root
    # print_tree(Parse)

    # Extracting the body
    body = Parse.body

    # Extracting all paragraphs within the body
    paragraphs = body.find_all('p') if body else []

    # Print the text of each paragraph within the body
    article_text = ""
    for i, para in enumerate(paragraphs):
        #print(f"Paragraph {i+1}: {para.get_text()}")
        #print(f"{para.get_text()}")
        article_text += para.get_text()

    #print(article_text)
    all_text.append(article_text)

# %%
