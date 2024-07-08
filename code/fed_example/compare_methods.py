"""
Compares all methods with the human labels on the test set
"""

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score

# main paths
dict_path = "../../dictionaries/"
data_path = "../../data/"
output_path = "../../output/"
models_path = "../../models/"

# %%

# load original data
df_test = pd.read_csv(data_path + "fed_test.csv",
                      dtype={'sentiment': 'category'})

# create mapping from label to sentiment
label2sent = {l:sent for l,sent in enumerate(df_test['sentiment'].cat.categories)}

# get all files with predictions from the methods
files = os.listdir(output_path)
files = [f for f in files if "fed" in f]
all_names = []
for f in files:
    method_name = f.replace("fed_tagged_", "").replace(".csv", "")
    all_names.append(method_name)
    df_method = pd.read_csv(output_path + f)
    if df_method["prediction"].dtype == int:
        df_method["prediction"] = df_method["prediction"].apply(lambda x: label2sent[x])
    
    df_method.columns = ["ID", f"prediction_{method_name}"]
    df_test = pd.merge(df_test, df_method, on="ID")

# %%

# calculate the accuracy of each method
all_accuracies = {}
for method in all_names:
    all_accuracies[method] = accuracy_score(df_test["sentiment"], df_test[f"prediction_{method}"])

# add the number of parameters
all_num_params = {'dictionary': 1,
                'gpt-3.5-turbo-0125': 175000000000,
                'gpt-4o': 1000000000000,
                "gemini-1.5-flash": 600000000000,
                'lr': 1200,
                'distilbert-base-uncased': 66000000
                  }

all_clean_names = {'dictionary': 'Dictionary',
                'gpt-3.5-turbo-0125': 'GPT-3.5',
                'gpt-4o': "GPT-4o",
                "gemini-1.5-flash": "Gemini 1.5",
                'lr': "Logistic Regression",
                'distilbert-base-uncased': "DistilBert"
                  }


df_plot = pd.DataFrame([all_accuracies, all_num_params, all_clean_names]).T
df_plot.columns = ["accuracy", "num_params", "clean_name"]
df_plot["error_rate"] = 1 - df_plot['accuracy']
df_plot

#%%

# check which methods are dominated by others
all_dominations = []
for method1 in df_plot.index:
    print(f"================== Comparing {method1} =====================")
    dominated = False
    method1_data = df_plot.loc[method1]
    num_params1 = method1_data["num_params"]
    error_rate1 = method1_data["error_rate"]
    for method2 in df_plot.index:
        if method2 != method1:
            method2_data = df_plot.loc[method2]
            num_params2 = method2_data["num_params"]
            error_rate2 = method2_data["error_rate"]
            # compare the number of parameters
            if (num_params2 < num_params1) and (error_rate2 < error_rate1):
                print(f"{method1} is dominated by {method2}")
                dominated = True
                break
    if not dominated:
        print(f"{method1} is not dominated by any other method")
    
    all_dominations.append(dominated)
                
df_plot["dominated"] = all_dominations
df_plot

# %%


# Set the style and figure size
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# Create the scatter plot
ax = sns.scatterplot(data=df_plot, x='num_params', y='error_rate', s=100)

# Set x-axis to log scale
plt.xscale('log')

# Customize the plot
plt.title('Error Rate vs Number of Parameters', fontsize=16)
plt.xlabel('Number of Parameters (log scale)', fontsize=12)
plt.ylabel('Error Rate', fontsize=12)

# Add method names as annotations
for idx, row in df_plot.iterrows():
    plt.annotate(row['clean_name'], (row['num_params'], row['error_rate']), 
                 xytext=(5, 5), textcoords='offset points', fontsize=10)

# Adjust the plot layout
plt.tight_layout()

# Show the plot
plt.show()
# %%
