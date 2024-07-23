"""
Uses OpenAI's API to label text data.
"""

#%%

import pandas as pd
import openai
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import yaml
import json
import tiktoken

# main paths
data_path = "../../data/"
model_path = "../../models/"
output_path = "../../output/"
keys_path = "../../../keys/"    # of course not in the repo ;)

# read main parameters from file
with open("params.yaml") as stream:
    params = yaml.safe_load(stream)

#%%

with open(keys_path + 'openai_api.txt', 'r') as file:
    api_key = file.read()

# initialize a client using the API key
client = OpenAI(api_key=api_key)

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

#%%

for i, row in df_test.sample(10).iterrows():
    print(row["label"])
    print(row["text"])
    print("\n\n======================\n")

# %%

system_prompt = "You are an expert in analyzing news articles and carefully extracting information from them. You have an advanced degree in Economics."

def generation_prompt(text):
    prompt = f"""
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

# initialize a tokenizer to count number of tokens before sending request
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")

#%%

# process all examples
max_tokens = 16385
all_responses = []
success_json = []
for text in tqdm(df_test["text"]):
    my_prompt = generation_prompt(text)
    try:
        response = client.chat.completions.create(
            model=params["llm_openai_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": my_prompt},
            ],
            max_tokens=params["llm_max_tokens"],
            temperature=params["llm_temperature"]
        )
    except openai.BadRequestError:
        # TODO: reduce the size of the prompt
        tokens = tokenizer.encode(my_prompt)
        num_tokens = len(tokens)
        if num_tokens > max_tokens:
            my_prompt = my_prompt[:int(16385/5)]
            all_responses.append("")
            success_json.append(False)
            continue
    # get response    
    try:
        result = response.choices[0].message.content
    except:
        print("Couldnt extract content out of response")
        all_responses.append("")
        success_json.append(False)
        continue
    # transform into JSON
    try:
        if "```json" in result:
            result = parse_json_string(result)
        
        response_dict = json.loads(result)
        all_responses.append(response_dict)
        success_json.append(True)
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        all_responses.append(result)
        success_json.append(False)

#%%

# # append data to existing dataframe
# all_epu = [r["EPU"] for r in all_responses]
# df_test["prediction"] = all_epu
# df_test = df_test[["article", "prediction"]] 
# df_test.to_csv(output_path + f"epu_tagged_{params['llm_openai_model']}.csv", index=False)

# %%

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
df_save.to_csv(output_path + f"epu_tagged_{params['llm_openai_model']}.csv", index=False)

# %%
