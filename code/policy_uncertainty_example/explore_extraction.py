#%%

import pandas as pd
import numpy as np
import yaml
import os
import shutil

# main paths
data_path = "../../data/"

# %%

# load labels
df_labels = pd.read_csv(data_path + "epu_modern_labels.csv")

# load text
df_text = pd.read_csv(data_path + "policy_uncertainty_extracted.csv", index_col=0)
# %%
