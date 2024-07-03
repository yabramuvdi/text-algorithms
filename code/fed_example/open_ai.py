#%%

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import numpy as np

# main paths
data_path = "../data/"
model_path = "../models/"
output_path = "../output/"
#%%

with open('../keys/openai_api.txt', 'r') as file:
    # Read the content of the file
    api_key = file.read()

# initialize a client using the API key
client = OpenAI(api_key=api_key)

# %%

#============================
# FED
#============================

# read train and test data 
df_finetune = pd.read_csv(data_path + "fed_train.csv", 
                          dtype={'sentiment': 'category'})
df_test = pd.read_csv(data_path + "fed_test.csv",
                      dtype={'sentiment': 'category'})

num_categories = len(df_finetune["sentiment"].cat.categories)


# %%

def generation_prompt(text):
    prompt = f"""
    Your task is to classify the text into one of the three categories ("dovish", "neutral", "hawkish").
    The text is taken at random from the texts of FOMC announcements.
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

# process all examples
all_responses = []
for text in tqdm(df_test["text"]):
    my_prompt = generation_prompt(text)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",    # name of the model,
        messages=[
            {"role": "system", "content": "You are a research assistant working for the Fed. You have a degree in Economics."},
            {"role": "user", "content": my_prompt},
        ],
        max_tokens=15,               # max number of tokens to be generate
        temperature=0                # temperature
    )
    result = response.choices[0].message.content
    all_responses.append(classify_response(result))

# append data to existing dataframe
df_test["gpt_response"] = all_responses
df_test.to_csv(output_path + "fed_test_gpt_labels.csv", index=False)