""" Loads and prepares the Economic Policy Uncertainty Data
"""

#%%

import pandas as pd
import numpy as np
import yaml
import os
import shutil
from sklearn.model_selection import train_test_split

# main paths
data_path = "../../data/"

# read main parameters from file
with open("params.yaml") as stream:
    params = yaml.safe_load(stream)

seed = params["seed"]
test_size = params["test_size"]

# %%

#===============
# Join labels with text
#===============

# load complete master data
df_master = pd.read_csv(data_path + "epu_master_complete.csv")

# create article ID
df_master['article_id'] = df_master.apply(lambda row: row['unique_id_current'] if pd.notna(row['unique_id_current']) else row['article_number'], axis=1)
df_master["article_id"] = df_master["article_id"].astype(str)
df_master.drop(columns=["article_number", "unique_id_current"], inplace=True)
df_master

# %%

# load text data
df_text = pd.read_parquet(data_path + "epu_text_complete.parquet")

#%%

df_complete = pd.merge(df_master, df_text, how="left", on=["vintage", "article_id"])
df_complete.dropna(inplace=True)
df_complete.reset_index(inplace=True, drop=True)

# %%

#===============
# Train-Test split
#===============

# perform a train/test split
df_train, df_test = train_test_split(df_complete, 
                                     stratify=df_complete['vintage'],
                                     test_size=test_size, 
                                     random_state=seed)


#%%

# keep multiple labels per article for training
df_train.rename(columns={"EPU": "label"}, inplace=True)
df_train.drop(columns=["num_words"], inplace=True)
df_train.to_parquet(data_path + "epu_train.parquet", index=False)

#%%

# get unique label as a majority vote for testing
df_test_unique = df_test[["vintage", "article_id", "EPU"]].groupby(["vintage", "article_id"], as_index=False).mean()
df_test_text = df_test.drop_duplicates()

def classify_epu(score):
    if score > 0.5:
        return 1
    elif score < 0.5:
        return 0
    else:
        return np.nan
    
df_test_unique["label"] = df_test_unique["EPU"].apply(classify_epu)
df_test_unique.dropna(inplace=True)
df_test_unique.drop(columns=["EPU"], inplace=True)

#%%

df_test_final = pd.merge(df_test_unique, df_test_text, how="left", on=["vintage", "article_id"])
df_test_final.drop(columns=["num_words"], inplace=True)

#%%

df_test_final.to_parquet(data_path + "epu_test.parquet", index=False)

# %%

#===============
# Manual verification
#===============

for i, row in df_test_final.sample(10).iterrows():
    print(row["vintage"], row["article_id"])
    print(row["year"], row["month"])
    print(row["text"])
    print("\n=============================\n")
# %%
