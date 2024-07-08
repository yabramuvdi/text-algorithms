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

#### read test data
df_test = pd.read_csv(data_path + "fed_test.csv",
                      dtype={'sentiment': 'category'})

#%%

#### hawkish dicitonary
hawk_dict = pd.read_csv(dict_path + "hawkish.csv")
hawk_dict["part"] = hawk_dict["part"].astype(bool)
hawk_dict

#%%

my_hawk_dict = Dictionary(list(hawk_dict["term"].values), 
                          list(hawk_dict["part"].values),
                          flexible_multi_word=True,
                          search_type="all",
                          return_matches=True,
                          ignore_case=True
                         )
          
my_hawk_dict.gen_dict_regex()
my_hawk_dict.dict_regex

#%%

# tag all text from the dataframe
results = df_test["text"].apply(my_hawk_dict.tag_text)
hawkish_boolean = [match[0] for match in results]
hawkish_terms = [match[1] for match in results]
df_test["hawkish"] = hawkish_boolean
df_test["hawkish_matches"] = hawkish_terms
df_test["num_hawkish"] = df_test["hawkish_matches"].apply(len)
df_test

#%%

#### dovish dicitonary
dove_dict = pd.read_csv(dict_path + "dovish.csv")
dove_dict["part"] = dove_dict["part"].astype(bool)
dove_dict

# %%

my_dove_dict = Dictionary(list(dove_dict["term"].values), 
                          list(dove_dict["part"].values),
                          flexible_multi_word=True,
                          search_type="all",
                          return_matches=True,
                          ignore_case=True
                         )
          
my_dove_dict.gen_dict_regex()
my_dove_dict.dict_regex

#%%

# apply to all text of a pandas dataframe
results = df_test["text"].apply(my_dove_dict.tag_text)
dovish_boolean = [match[0] for match in results]
dovish_terms = [match[1] for match in results]
df_test["dovish"] = dovish_boolean
df_test["dovish_matches"] = dovish_terms
df_test["num_dovish"] = df_test["dovish_matches"].apply(len)
df_test

# %%

#### final dictionary measure
def categorize(row):
    if row["num_hawkish"] > row["num_dovish"]:
        return "hawkish"
    elif row["num_hawkish"] < row["num_dovish"]:
        return "dovish"
    else:
        return "neutral"

df_test["dictionary"] = df_test.apply(lambda x: categorize(x), axis=1)
# %%

#### calculate accuracy
df_test["dict_correct"] = df_test["dictionary"] == df_test["sentiment"]
dict_accuracy = df_test["dict_correct"].mean()
print(f"Random guessing accuracy: {1/3}")
print(f"Accuracy of Dictionary: {dict_accuracy}")

# %%

#### save results
df_test = df_test[["ID", "dictionary"]] 
df_test.columns = ["ID", "prediction"]
df_test.to_csv(output_path + "fed_tagged_dictionary.csv", index=False)

# %%
