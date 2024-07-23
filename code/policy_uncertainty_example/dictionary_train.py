"""
Implements a dictionary-based approach using regular expressions to
search for the presence of certain terms in the text. Assigns a label
to the text based on the results of the dictionary search.
"""

#%%

import pandas as pd
import re
import string

# import custom modules
import sys
sys.path.insert(1, '../utils/')
from dictionary_methods import Dictionary
from utils import clean_sequence

# main paths
dict_path = "../../dictionaries/"
data_path = "../../data/"
output_path = "../../output/"

#%%

#======================
# Data
#======================

# read test data
df_train = pd.read_csv(data_path + "epu_train.csv")

# read text data
df_text = pd.read_csv(data_path + "epu_modern_text.csv", index_col=0)

# join data
df_train = pd.merge(df_train, df_text, how="left", 
                   left_on="unique_id_current",
                   right_on="article")

#%%

#======================
# Policy dicitonary
#======================

policy_dict = pd.read_csv(dict_path + "policy_terms.csv")
policy_dict["part"] = policy_dict["part"].astype(bool)
policy_dict

#%%

my_policy_dict = Dictionary(list(policy_dict["term"].values), 
                          list(policy_dict["part"].values),
                          flexible_multi_word=True,
                          search_type="all",
                          return_matches=True,
                          ignore_case=True
                         )

my_policy_dict.gen_dict_regex()
my_policy_dict.dict_regex

#%%

# tag all text from the dataframe
results = df_train["text"].apply(my_policy_dict.tag_text)
policy_boolean = [match[0] for match in results]
policy_terms = [match[1] for match in results]
df_train["policy"] = policy_boolean
df_train["policy_matches"] = policy_terms
df_train["num_policy"] = df_train["policy_matches"].apply(len)
df_train

#%%

#======================
# Economic terms dicitonary
#======================

econ_dict = pd.read_csv(dict_path + "economy_terms.csv")
econ_dict["part"] = econ_dict["part"].astype(bool)
econ_dict

# %%

my_econ_dict = Dictionary(list(econ_dict["term"].values), 
                          list(econ_dict["part"].values),
                          flexible_multi_word=True,
                          search_type="all",
                          return_matches=True,
                          ignore_case=True
                         )
          
my_econ_dict.gen_dict_regex()
my_econ_dict.dict_regex

#%%

# apply to all text of a pandas dataframe
results = df_train["text"].apply(my_econ_dict.tag_text)
econ_boolean = [match[0] for match in results]
econ_terms = [match[1] for match in results]
df_train["econ"] = econ_boolean
df_train["econ_matches"] = econ_terms
df_train["num_econ"] = df_train["econ_matches"].apply(len)
df_train

# %%

#======================
# Uncertainty terms dicitonary
#======================

uncertain_dict = pd.read_csv(dict_path + "uncertainty_terms.csv")
uncertain_dict["part"] = uncertain_dict["part"].astype(bool)
uncertain_dict

# %%

my_uncertain_dict = Dictionary(list(uncertain_dict["term"].values), 
                          list(uncertain_dict["part"].values),
                          flexible_multi_word=True,
                          search_type="all",
                          return_matches=True,
                          ignore_case=True
                         )
          
my_uncertain_dict.gen_dict_regex()
my_uncertain_dict.dict_regex

#%%

# apply to all text of a pandas dataframe
results = df_train["text"].apply(my_uncertain_dict.tag_text)
uncertain_boolean = [match[0] for match in results]
uncertain_terms = [match[1] for match in results]
df_train["uncertain"] = uncertain_boolean
df_train["uncertain_matches"] = uncertain_terms
df_train["num_uncertain"] = df_train["uncertain_matches"].apply(len)
df_train

# %%

#======================
# Final measure (EPU)
#======================

def categorize(row):
    if row["policy"] == row["econ"] == row["uncertain"] == True:
        return 1
    else:
        return 0

df_train["dictionary"] = df_train.apply(lambda x: categorize(x), axis=1)

# %%

# calculate dictionary accuracy
dict_acc = (df_train["label"] == df_train["dictionary"]).mean()
print(f"Dictionary accuracy: {dict_acc}")

#%%

# do some readings
for i, row in df_train.loc[df_train["dictionary"] ==1].sample(1).iterrows():
    print(row["label"])
    print(row["text"])

# %%

#### save results
df_train = df_train[["article", "dictionary"]] 
df_train.columns = ["article", "prediction"]
df_train.to_csv(output_path + "epu_train_tagged_dictionary.csv", index=False)

# %%

#======================
# Consolidate train and test data into one file (for Stephen)
#======================

# read train data
df_train = pd.read_csv(data_path + "epu_train.csv")

# read test data
df_test = pd.read_csv(data_path + "epu_test.csv")

# join the two datasets
df = pd.concat([df_train, df_test])

#%%

# read text data
df_text = pd.read_csv(data_path + "epu_modern_text.csv", index_col=0)

# join data with text
df = pd.merge(df, df_text, how="left", 
              left_on="unique_id_current",
              right_on="article")

df.drop(columns=["unique_id_current"], inplace=True)
# %%

# join data with dictionary labels
df_train_dict = pd.read_csv(output_path + "epu_train_tagged_dictionary.csv")
df_train_dict.columns = ["article", "dict_label"]

df_test_dict = pd.read_csv(output_path + "epu_tagged_dictionary.csv")
df_test_dict.columns = ["article", "dict_label"]

df_dict = pd.concat([df_train_dict, df_test_dict])
df_dict = df_dict.groupby("article", as_index=False).mean()
# %%

df_save = pd.merge(df, df_dict, how="left", on="article")
df_save.rename(columns={"label": "human_label"}, inplace=True)
df_save.to_parquet(output_path + "epu_tagged.parquet", index=False)
# %%
