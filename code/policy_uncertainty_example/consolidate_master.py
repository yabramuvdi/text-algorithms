""" Loads labels from the Economic Policy Uncertainty Data and
    creates a single file with labels and identifiers for the articles
"""

#%%

import pandas as pd
import numpy as np

# main paths
data_path = "../../data/"


# %%

#===============
# Master data with labels
#===============

# load complete master data
df_master = pd.read_stata(data_path + "EPU_AUDIT_MASTER_FILE.dta")

# restric only to modern articles
# df_master = df_master.loc[df_master["vintage"] == "Current"]

# simplify
df_master = df_master[["vintage", "article_number", "year", "month", "EPU"]]
#%%

# load master data for modern articles (provided by Baker)
df_master_modern = pd.read_stata(data_path + "modern_unique_ids.dta")
df_master_modern.drop_duplicates(inplace=True)

# %%

# join with the rest of the labels
df_complete = pd.merge(df_master, df_master_modern, how="left", on=["article_number", "year", "month", "vintage"])

# %%

# save
df_complete.to_csv(data_path + "epu_master_complete.csv", index=False)
# %%
