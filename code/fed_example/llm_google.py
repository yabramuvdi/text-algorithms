"""
Uses Google's API to label text data.
"""

#%%

import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import numpy as np
import yaml

# main paths
data_path = "../../data/"
model_path = "../../models/"
output_path = "../../output/"
keys_path = "../../../keys/"    # of course not in the repo ;)

# read main parameters from file
with open("params.yaml") as stream:
    params = yaml.safe_load(stream)

# load api keys
with open(keys_path + 'google_api.txt', 'r') as file:
    api_key = file.read()


# %%

# read train and test data 
df_finetune = pd.read_csv(data_path + "fed_train.csv", 
                          dtype={'sentiment': 'category'})
df_test = pd.read_csv(data_path + "fed_test.csv",
                      dtype={'sentiment': 'category'})

num_categories = len(df_finetune["sentiment"].cat.categories)


# %%

#===============
# Prompt
#===============

def generation_prompt(text):
    prompt = f"""
    You are a research assistant working for the Fed. You have a degree in Economics.

    Your task is to classify the text into one of the three categories ("dovish", "neutral", "hawkish").
    The text is taken at random from multiple FOMC announcements.
    Provide your output in json format with a key "category" and the selected category.

    Text:
    {text}
    """
    return prompt

def classify_response(result):
    if "hawkish" in result:
        return "hawkish"
    elif "dovish" in result:
        return "dovish"
    elif "neutral" in result:
        return "neutral"
    else:
        return np.nan

#%%
      
#===============
# Load model
#===============

genai.configure(api_key=api_key)

# Set up the model
generation_config = {
"temperature": params["llm_temperature"],
"top_p": 1,
"top_k": 1,
"max_output_tokens": params["llm_max_tokens"],
}

# initialize the model
model = genai.GenerativeModel(model_name=params["llm_google_model"],
                              generation_config=generation_config)

#%%

#===============
# Process data
#===============

# process all examples
all_responses = []
for text in tqdm(df_test["text"]):
    my_prompt = generation_prompt(text)
    response = model.generate_content(my_prompt)
    result = response.text
    all_responses.append(classify_response(result))

#%%
    
# append data to existing dataframe and save
df_test["prediction"] = all_responses
df_test = df_test[["ID", "prediction"]] 
df_test.to_csv(output_path + f"fed_tagged_{params['llm_google_model']}.csv", index=False)
