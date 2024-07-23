"""
Uses Google's API to label text data.
"""

#%%

import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import numpy as np
import yaml
import json

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

# read text data
df_text = pd.read_csv(data_path + "epu_modern_text.csv", index_col=0)

# read test labels
df_test = pd.read_csv(data_path + "epu_test.csv")
df_test["label"] = df_test["label"].astype(int) 

# join data
df_test = pd.merge(df_test, df_text, how="left", 
                   left_on="unique_id_current",
                   right_on="article")

# %%

#===============
# Prompt
#===============

def generation_prompt(text):
    prompt = f"""
    You are an expert in analyzing news articles and carefully extracting information from them. You have an advanced degree in Economics.

    Your task is to analyze a news article and identify if it is about policy-related aspects of economic uncertainty, even if only to a limited extent. If it is you should output EPU=1 and if it is not you should output EPU=0.
    Take into account that:
    - The article need not contain extensive remarks about policy-related aspects of economic uncertainty, nor be mainly about economic policy uncertainty to be coded as EPU=1.
    - However, if the article discusses economic uncertainty in one part and policy in another part but never discusses policy in connection to economic uncertainty, then do not code it as about economic policy uncertainty.

    Provide your output in JSON format with a key "EPU" and a 1 or a 0.

    Article:
    {text}
    """
    return prompt

# auxiliary function to clean the output
def parse_json_string(json_string):
    clean_string = json_string.replace('```json\n', '').replace('\n```', '')
    return clean_string

#%%
      
#===============
# Load model
#===============

genai.configure(api_key=api_key)

# Set up the model
generation_config = {
"temperature": params["llm_temperature"],
# "top_p": 1,
# "top_k": 1,
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
success_json = []
for text in tqdm(df_test["text"]):
    # generate prompt and call the model
    my_prompt = generation_prompt(text)
    response = model.generate_content(my_prompt)
    # extract response
    try:
        result = response.text
    except ValueError:
        all_responses.append("")
        success_json.append(False)
        continue
    # transform into JSON
    try:
        clean_result = parse_json_string(result)
        response_dict = json.loads(clean_result)
        all_responses.append(response_dict)
        success_json.append(True)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        all_responses.append(result)
        success_json.append(False)

#%%


#### append data to existing dataframe and save

# Define the keys for the dictionaries
keys = all_responses[0].keys()
# Replace non-dictionary elements with dictionaries with empty values
all_responses_clean = [item if isinstance(item, dict) else {key: '' for key in keys} for item in all_responses]
# create a DataFrame from the new data
new_data_df = pd.DataFrame(all_responses_clean)
new_data_df.columns = ["prediction"]

#%%

df_save = df_test.join(new_data_df)
df_save = df_save[["article", "prediction"]] 
df_save.to_csv(output_path + f"epu_tagged_{params['llm_google_model']}.csv", index=False)

# %%
