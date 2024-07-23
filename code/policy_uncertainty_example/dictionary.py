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

# import custom module
import sys
sys.path.insert(1, '../utils/')
from utils import clean_sequence

#%%

#======================
# Data
#======================

# # read test data
# df_test = pd.read_csv(data_path + "epu_test.csv")

# # read text data
# df_text = pd.read_csv(data_path + "epu_modern_text.csv", index_col=0)

# # join data
# df_test = pd.merge(df_test, df_text, how="left", 
#                    left_on="unique_id_current",
#                    right_on="article")

df_test = pd.read_parquet(data_path + "epu_test.parquet")

# define punctuation symbols to remove
punctuation = string.punctuation
punctuation = punctuation.replace("-", "")
punctuation = punctuation.replace("'", "")

df_test["text"] = df_test["text"].apply(lambda x: clean_sequence(x, punctuation))
df_test

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

test = "Fiscal policy in the white house is regulation federal Reserve"
my_policy_dict.tag_text(test)

#%%

# tag all text from the dataframe
results = df_test["text"].apply(my_policy_dict.tag_text)
policy_boolean = [match[0] for match in results]
policy_terms = [match[1] for match in results]
df_test["policy"] = policy_boolean
df_test["policy_matches"] = policy_terms
df_test["num_policy"] = df_test["policy_matches"].apply(len)
df_test

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
results = df_test["text"].apply(my_econ_dict.tag_text)
econ_boolean = [match[0] for match in results]
econ_terms = [match[1] for match in results]
df_test["econ"] = econ_boolean
df_test["econ_matches"] = econ_terms
df_test["num_econ"] = df_test["econ_matches"].apply(len)
df_test

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
results = df_test["text"].apply(my_uncertain_dict.tag_text)
uncertain_boolean = [match[0] for match in results]
uncertain_terms = [match[1] for match in results]
df_test["uncertain"] = uncertain_boolean
df_test["uncertain_matches"] = uncertain_terms
df_test["num_uncertain"] = df_test["uncertain_matches"].apply(len)
df_test

# %%

#======================
# Final measure (EPU)
#======================

def categorize(row):
    if row["policy"] == row["econ"] == row["uncertain"] == True:
    #if row["policy"] == True:
    
        return 1
    else:
        return 0

df_test["dictionary"] = df_test.apply(lambda x: categorize(x), axis=1)

# %%

# calculate dictionary accuracy
dict_acc = (df_test["label"] == df_test["dictionary"]).mean()
print(f"Dictionary accuracy: {dict_acc}")

#%%

# do some readings
for i, row in df_test.loc[df_test["dictionary"] ==1].sample(1).iterrows():
    print(row["label"])
    print(row["text"])

# %%

#### save results
df_test = df_test[["article_id", "vintage", "dictionary"]] 
df_test.columns = ["article_id", "vintage", "prediction"]
df_test.to_csv(output_path + "epu_tagged_dictionary.csv", index=False)

# %%
