"""
Uses Anthorpic's API to label text data.
"""
#%%

import pandas as pd
from tqdm import tqdm
import numpy as np
import yaml
from anthropic import Anthropic

# main paths
data_path = "../../data/"
model_path = "../../models/"
output_path = "../../output/"
keys_path = "../../../keys/"    # of course not in the repo ;)

# read main parameters from file
with open("params.yaml") as stream:
    params = yaml.safe_load(stream)

#%%

with open(keys_path + 'anthropic_api.txt', 'r') as file:
    api_key = file.read()

client = Anthropic(api_key=api_key)

#%%

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
    The text is taken at random from multiple FOMC announcements.
    Provide your output in JSON format with a key "category" and the selected category.

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
    response = client.messages.create(
        model=params["llm_anthropic_model"],
        max_tokens=100,
        temperature=params["llm_temperature"],
        system="You are a research assistant working for the Fed. You have a degree in Economics. Reply only with a JSON file.",
        messages=[
            {"role": "user",
             "content": my_prompt,
            },
            ],
        )
    result = response.content[0].text
    all_responses.append(classify_response(result))

#%%
    
# append data to existing dataframe
df_test["prediction"] = all_responses
df_test = df_test[["ID", "prediction"]] 
df_test.to_csv(output_path + f"fed_tagged_{params['llm_anthropic_model']}.csv", index=False)

#%%
