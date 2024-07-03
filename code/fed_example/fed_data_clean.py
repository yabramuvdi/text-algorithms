#%%

import pandas as pd
import string
from sklearn.model_selection import train_test_split
import yaml

# import custom module
import sys
sys.path.insert(1, '../utils/')
from utils import clean_sequence

# main paths
dict_path = "../../dictionaries/"
data_path = "../../data/"

# read main parameters from file
with open("params.yaml") as stream:
    params = yaml.safe_load(stream)

seed = params["seed"]
test_size = params["test_size"]

#%%

# read data and apply basic cleaning
df = pd.read_csv(data_path + "FED_classification.csv", sep="\t")

# define punctuation symbols to remove
punctuation = string.punctuation
punctuation = punctuation.replace("-", "")
punctuation = punctuation.replace("'", "")

df["text_clean"] = df["text"].apply(lambda x: clean_sequence(x, punctuation))
df

#%%

# transform the label into a category
df['sentiment'] = df['sentiment'].astype('category')
df['label'] = df['sentiment'].cat.codes

num_categories = len(df["sentiment"].cat.categories)
#%%

# create list with all the indexes of available sentences
idxs = list(df.index)

# perform a train/test split
train_idxs, test_idxs = train_test_split(idxs, 
                                         test_size=test_size, 
                                         random_state=seed)

df_finetune = df.loc[train_idxs].copy()
df_test = df.loc[test_idxs].copy()

df_finetune.to_csv(data_path + "fed_train.csv", index=False)
df_test.to_csv(data_path + "fed_test.csv", index=False)