#=============================================
# Script to implement a supervised learning approach
# to predict remote work from a document term matrix
#=============================================

#%%

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import l1_min_c
import pickle
from joblib import dump, load

import time
import string
import re
import matplotlib.pyplot as plt

from utils import clean_sequence


# paths and params
seed = 92
n_splits = 5

# main paths
dict_path = "../dictionaries/"
data_path = "../data/"
output_path = "../output/"

#%%

#============================
# FED
#============================

# read train and test data 
df_train = pd.read_csv(data_path + "fed_train.csv", 
                          dtype={'sentiment': 'category'})
df_test = pd.read_csv(data_path + "fed_test.csv",
                      dtype={'sentiment': 'category'})
# %%

#=============================
# 0. Read data and clean text
#=============================


# # define the columns that have the dictionary terms
# dict_cols = ['working remotely', 'working from home', 'work remotely',
#              'work from home', 'work at home', 'teleworking', 'telework',
#              'telecommuting', 'telecommute', 'smartworking', 
#              'smart working', 'remote work teleworking', 
#              'remote work', 'remote', "remotely", 'homeoffice',
#              'home office', 'home based', "homebased"]

#%%

#=============================
# 1. Tokenization
#=============================

pattern = r'''
          (?x)                # set flag to allow verbose regexps (to separate logical sections of pattern and add comments)
          \w+(?:-\w+)*        # preserve expressions with internal hyphens as single tokens
          |\b\w+\b            # single letter words
          '''

# create a CountVectorizer object straight from the raw text
count_vectorizer = CountVectorizer(encoding='utf-8',
                                   token_pattern=pattern,
                                   lowercase=True,
                                   strip_accents="ascii",
                                   stop_words="english", 
                                   ngram_range=(1, 1),       # generate only unigrams
                                   analyzer='word',          # analysis at the word-level
                                   max_features=5000,        # we could impose a maximum number of vocabulary terms
                                   )
                                   
# transform our sequences into a document-term matrix
dt_matrix_train = count_vectorizer.fit_transform(df_train["text_clean"])
dt_matrix_train = dt_matrix_train.toarray()
print(f"Document-term matrix created with shape: {dt_matrix_train.shape}")

#%%

# explore vocab
vocab = count_vectorizer.vocabulary_
n_terms = len(vocab)

# save vectorizer
dump(count_vectorizer, output_path + 'vectorizer.joblib') 

#%%

# # i) replace the columns for terms that are in the vocabulary
# # ii) append the columns from any term in the dictionary that is not in the vocab
# dict_cols_idx = {}
# for col in dict_cols:
    
#     if col in vocab.keys():
#         # print(f"Replacing column: {col} in dt matrix")
#         # # get the dictionary data for the term
#         # col_vector = df_train[col].values.astype(int)
#         # dt_matrix_train[:, vocab[col]] = col_vector
#         pass

#     if col not in vocab.keys():
#         print(f"Adding column: {col} to dt matrix")
#         col_vector = df_train[col].values.astype(int)
#         #print(f"Average number of hits: {np.mean(col_vector)}")
        
#         col_vector = np.expand_dims(col_vector, axis=1)
#         dt_matrix_train = np.hstack((dt_matrix_train, col_vector))
#         print(f"New size of dt matrix: {dt_matrix_train.shape}")

#         # store position of term in the matrix
#         dict_cols_idx[dt_matrix_train.shape[1]-1] = col

# print(f"Final size of dt matrix: {dt_matrix_train.shape}")
#%%

# transform document-term matrix into binary form
dt_matrix_train_b = np.where(dt_matrix_train > 0, 1, dt_matrix_train)
dt_matrix_train_b.shape

#%%

#=============================
# 2. Prepare K-Fold splits and parameter Grid
#=============================

# C --> Inverse of regularization strength; must be a positive float. 
# Like in support vector machines, smaller values specify stronger regularization. 

#cs = l1_min_c(dt_matrix_train_b, df_train["labels"].values, loss="log") * np.logspace(0, 7, 100)
#cs = l1_min_c(dt_matrix_train_b, df_train["labels"].values, loss="log") * np.logspace(-3, 8, 5)
cs = np.linspace(-3, 3, 50)
cs = np.array([10**c for c in cs])
# plot lambdas
print(f"Min lambda: {np.min(cs)}, Max lambda: {np.max(cs)}")
plt.figure(figsize=(12,8))
plt.scatter(cs, cs)
plt.show()

# create grid with parameters
grid = {"penalty": ["l1"],
        "tol": [1e-4], 
        "C": 1/cs, 
        "fit_intercept": [True],  
        "random_state": [92],
        "solver": ["liblinear"], 
        "max_iter": [100]
        }

#%%

lr_cv = GridSearchCV(estimator=LogisticRegression(), 
                     param_grid=grid, 
                     cv=n_splits,
                     scoring="accuracy",
                     verbose=0,
                     n_jobs=-3)

lr_cv.fit(dt_matrix_train_b, df_train["label"].values)

print("\n=========================\n\n")
print("Tuned hpyerparameters : \n", lr_cv.best_params_)
print("Best F1 :", lr_cv.best_score_)
df_cv = pd.DataFrame(lr_cv.cv_results_)
df_cv.to_csv(output_path + "lr_cv_log_results_final.csv", index=False)

# plot results
plt.figure(figsize=(12,8))
plt.plot(df_cv["param_C"].values, df_cv["mean_test_score"].values)
plt.ylabel("F1 score")
plt.xlabel("Inverse regularization param (log)")
plt.xscale('log')
plt.show()

plt.figure(figsize=(12,8))
plt.plot(1/df_cv["param_C"].values, df_cv["mean_test_score"].values)
plt.ylabel("F1 score")
plt.xlabel("Regularization param (log)")
plt.xscale('log')
plt.show()

#%%

#=============================
# 3. Out of sample predictions
#=============================

# extract optimal model fitted on whole training data
lr_opt = lr_cv.best_estimator_

# save optimal model
dump(lr_opt, output_path + 'logistic_regression_opt.joblib') 

# %%

#=============================
# Preprocessing
#=============================

# # load data and select only relevant columns
# df_test = pd.read_csv(input_path + "dict_replication_test_sequences.csv")
# df_test["labels"] = df_test["labels"].astype(int)
# print(f"{len(df_test)} Test examples")

# # apply preprocessing
# df_test["clean_seq"] = df_test["sequence.x"].apply(lambda x: clean_sequences(x, punctuation))

# # define the columns that have the dictionary terms
# dict_cols = ['working remotely', 'working from home', 'work remotely',
#              'work from home', 'work at home', 'teleworking', 'telework',
#              'telecommuting', 'telecommute', 'smartworking', 
#              'smart working', 'remote work teleworking', 
#              'remote work', 'remote', "remotely", 'homeoffice',
#              'home office', 'home based', "homebased"]

# #%%

# # read some examples
# i = np.random.randint(0, len(df_test))
# print(df_test.loc[i, "narrow_result"])
# for col in dict_cols:
#     if df_test.loc[i, col] == True:
#         print(col)

# print("----------------------")
# print(df_test.loc[i, "sequence.x"])
# print("-------------------------")
# print(df_test.loc[i, "clean_seq"])

# %%

#=============================
# Tokenization
#=============================

# transform our sequences into a document-term matrix
# use the same vectorizer from the train data
dt_matrix_test = count_vectorizer.transform(df_test["text_clean"])
dt_matrix_test = dt_matrix_test.toarray()
print(f"Document-term matrix created with shape: {dt_matrix_test.shape}")

# #%%

# # i) replace the columns for terms that are in the vocabulary
# # ii) append the columns from any term in the dictionary that is not in the vocab
# for col in dict_cols:
    
#     if col in vocab.keys():
#         # print(f"Replacing column: {col} in dt matrix")
#         # # get the dictionary data for the term
#         # col_vector = df_test[col].values.astype(int)
#         # dt_matrix_test[:, vocab[col]] = col_vector
#         pass

#     if col not in vocab.keys():
#         print(f"Adding column: {col} to dt matrix")
#         col_vector = df_test[col].values.astype(int)
#         col_vector = np.expand_dims(col_vector, axis=1)
#         dt_matrix_test = np.hstack((dt_matrix_test, col_vector))
#         print(f"New size of dt matrix: {dt_matrix_test.shape}")

# print(f"Final size of dt matrix: {dt_matrix_test.shape}")
#%%

# transform document-term matrix into binary form
dt_matrix_test_b = np.where(dt_matrix_test > 0, 1, dt_matrix_test)
dt_matrix_test_b.shape

# %%

#=============================
# Prediction
#=============================

y_hat_test = lr_opt.predict(dt_matrix_test_b)
df_test["lr_pred"] = y_hat_test

#%%

#df_test.to_csv(output_path + "v1_lr_replication_predictions.csv", index=False)
print("Accuracy score in test data: ", accuracy_score(df_test["label"], df_test["lr_pred"]))

# %%

# #=============================
# # Analyze results
# #=============================

# # create inverse vocab
# idx2word = {idx:word for word,idx in vocab.items()}

# # explore weigth importance
# weights = lr_opt.coef_
# print(weights.shape)  

# # %%

# # order weigths
# n = 25
# ordered_idxs = np.argsort(weights)[0]
# ordered_weigths = np.take(weights, ordered_idxs)
# top_n = list(ordered_idxs[-n:])
# top_n.reverse()
# bottom_n = list(ordered_idxs[0:n])

# zeros = np.where(weights == 0)[1]
# print(f"Number of coefficients equal to zero: {len(zeros)}")
# print(f"Percentage of coefficients equal to zero: {np.round(len(zeros)/weights.shape[1], 3)*100}%")

# # %%

# # top words
# print("Top 25 coefficients (ranked)")
# for i, w in enumerate(top_n):
#     #print(idx2word[w], weights[0][w])
#     if w in idx2word.keys():
#         print(f"{i+1}. {idx2word[w]} ({np.round(weights[0][w], 2)})" )
#     # else:
#     #     print(f"{i+1}. {dict_cols_idx[w]} ({np.round(weights[0][w], 2)})" )

# print("\n=======================\n")

# # bottom words
# print("Bottom 25 coefficients (ranked)")
# for i, w in enumerate(bottom_n):
#     #print(idx2word[w], weights[0][w])
#     if w in idx2word.keys():
#         print(f"{i+1}. {idx2word[w]} ({np.round(weights[0][w], 2)})" )
#     # else:
#     #     print(f"{i+1}. {dict_cols_idx[w]} ({np.round(weights[0][w], 2)})" )

    
# # %%