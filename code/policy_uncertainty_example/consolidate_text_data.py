""" Joins all text data from different sources
"""

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# main paths
data_path = "../../data/"

# %%

#===============
# Data with text
#===============

# load modern text data
df_text_modern = pd.read_parquet(data_path + "epu_modern_text.parquet")
df_text_modern.dropna(inplace=True)
df_text_modern["vintage"] = "Current"
df_text_modern.rename(columns={"article": "article_num"}, inplace=True)

# load midcentury text data
df_text_mc = pd.read_parquet(data_path + "epu_midcentury_text.parquet")
df_text_mc.dropna(inplace=True)
df_text_mc["vintage"] = "Midcentury"

# load historical text data
df_text_hist = pd.read_parquet(data_path + "epu_historical_text.parquet")
df_text_hist.dropna(inplace=True)
df_text_hist["vintage"] = "Historical"

# load historical oversample text data
df_text_hist_over = pd.read_parquet(data_path + "epu_historical_oversample_text.parquet")
df_text_hist_over.dropna(inplace=True)
df_text_hist_over["vintage"] = "Hisotrical oversample"

# join all data
df_text = pd.concat([df_text_modern, df_text_mc, df_text_hist, df_text_hist_over])
df_text.columns = ["article_id", "text", "vintage"]
df_text
# %%

#===============
# Check number of "tokens"
#===============

# simple split using white spaces
df_text["num_words"] = df_text["text"].apply(lambda x: len(x.split()))

#%%

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(df_text['num_words'], edgecolor='black')
plt.title('Distribution of Number of Words')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.xticks(range(1, df_text['num_words'].max() + 2))
plt.grid(axis='y')

plt.show()
# %%

# drop any article with very few words
min_words = 100
df_text = df_text.loc[df_text["num_words"] > min_words]

#%%

# save data
df_text["article_id"] = df_text["article_id"].astype(str)
df_text.to_parquet(data_path + "epu_text_complete.parquet", index=False)
# %%
