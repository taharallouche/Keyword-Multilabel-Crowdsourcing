"""
This is the main file where the data (available at https://github.com/CrowdTruth/Events-in-videos) is loaded and prepared,
and the functions are executed.
The first part will consist of:
   -Generating simulated votes according to the noise model for different parameters.
   -Applying different aggregation rules on every instance of the dataset (5521), namely: the majority rule, the modal rule,
    the MLE with uniform prior and the MLE.
   -Computing, plotting and comparing the Hamming and 0/1 losses for each rule and parameters.

The second part will consist of:
    -Applying the the EM algorithm to estimate the groundtruth for different batches of the dataset.
    -Applying the simple majority rule.
    -Compute the Hamming and 0/1 accuracy for each method.
    -Compare them (using statistical tests in some cases).
    -Print the histograms of the estimated noise parameters.
"""

###################################################
######Importing libraries and function files#######
###################################################

import numpy as np
import pandas as pd
from Sim_fun import *
from Real_exp1 import *




###################################################
#############Preparing the dataset#################
###################################################

# Annotations dataframe
workersA = pd.read_csv('Data/_2_A_worker_vectors.csv', delimiter='|', index_col=False,
                       names=["worker_id", "item", "Basketball", "Children", "Gym class", "Graduation", "Other",
                              "Soccer", "Birthday", "Beach", "Parade", "Guitar", "Rain", "Fireworks",
                              "Floating lanterns", "Cake", "Sleeping", "Birthday drinks", "Boat"])

# Groundtruth dataframe

#Uncomment for full labels (16)
# Keywords = pd.read_csv("Data/_0_ACTUALUSED_keywords.csv", index_col=False, names=["Keyword"])
#
# TrueLabels = pd.read_csv("Data/_1_keyword_labels_binary.csv", index_col=False, delim_whitespace=True, names=["Gym
# class", "Floating lanterns", "Basketball", "Parade", "Fireworks", "Graduation", "Sleeping", "Guitar", "Birthday",
# "Rain", "Cake", "Birthday drinks", "Soccer", "Beach", "Children", "Boat", "Bahamas", "Puppets", "Basketball game",
# "Soccer game", "Candles", "Chefs", "Football", "Culinary school", "Uncle sam", "Baseball", "Zombies",
# "Birthday ballons", "Helicopter", "Hospital", "Motorcycles", "Other"])
#
# # Make Groundtruth annotation dataframe
# GroundTruth = pd.concat([Keywords, TrueLabels], axis=1)
# workersA_col = ["Basketball", "Children", "Gym class", "Graduation", "Other",
#                 "Soccer", "Birthday", "Beach", "Parade", "Guitar", "Rain", "Fireworks",
#                 "Floating lanterns", "Cake", "Sleeping", "Birthday drinks", "Boat"]

col = ["Basketball", "Children", "Gym class", "Graduation", "Other", "Soccer", "Birthday", "Parade", "Fireworks",
       "Cake"]

GroundTruth = pd.read_csv('Data/Ground.csv')

# GroundTruth Verification: Uncomment for groundtruth instance by instance verification
# instance =
# for keyword in GroundTruth.iloc[instance:, :]['Keyword']:
#     print(instance, " ", keyword, ": ")
#     instance += 1
#     val = input("OK?")
#     if val == "s":
#         break
#     for label in col:
#         print(label)
#         val = input(": ")
#         if val == "s":
#             break
#         else:
#             val = int(val == "a")
#             GroundTruth.loc[GroundTruth["Keyword"] == keyword, label] = val
#GroundTruth.to_csv(r'Data/Ground.csv', index = False)





###############################################
############ Descriptive analysis #############
###############################################

#Uncomment for correlation analysis
# Prior correlation
# import itertools
#
# prior_cor = pd.DataFrame(np.ones((10, 10)), index=GroundTruth[col].columns, columns=GroundTruth[col].columns)
# for col_pair in itertools.combinations(GroundTruth[col].columns, 2):
#     prior_cor.loc[col_pair] = prior_cor.loc[tuple(reversed(col_pair))] = jaccard_score(GroundTruth[col].iloc[:1200,][col_pair[0]],
#                                                                                        GroundTruth[col].iloc[:1200,][col_pair[1]])
#
# f = plt.figure()
# plt.matshow(prior_cor, fignum=f.number)
# plt.xticks(range(prior_cor.shape[1]), col, fontsize=8, rotation=45)
# plt.yticks(range(prior_cor.shape[1]), col, fontsize=8)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=6)
# plt.title('Groundtruth Jaccard Similarity between Labels', fontsize=8)
# plt.savefig('Groundtruth Jaccard Similarity between Labels.png')
#
# for colu in col:
#     workersA.loc[workersA[colu]== 2,colu]=0
#
# bool_cols = [colu for colu in workersA[col]
#              if np.isin(workersA[col][colu].dropna().unique(), [0, 1]).all()]
# # Votes correlation
# vote_cor = pd.DataFrame(np.ones((10, 10)), index=workersA[col].columns, columns=workersA[col].columns)
# for col_pair in itertools.combinations(col, 2):
#     vote_cor.loc[col_pair] = vote_cor.loc[tuple(reversed(col_pair))] = jaccard_score(workersA[col][col_pair[0]].to_numpy(),
#                                                                                      workersA[col][col_pair[1]].to_numpy())
#
# f = plt.figure()
# plt.matshow(vote_cor, fignum=f.number)
# plt.xticks(range(vote_cor.shape[1]), col, fontsize=8, rotation=45)
# plt.yticks(range(vote_cor.shape[1]), col, fontsize=8)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=6)
# plt.title('Votes Jaccard Similarity between Labels', fontsize=8)
# plt.savefig('Votes Jaccard Similarity between Labels.png')
# votes_correlation = workersA[col].corr(method="pearson")

#####################################
######## Part I: Simulations ########
#####################################

# Real labels, Synthetic votes
GroundTruth_data = GroundTruth[col].to_numpy()
n_batch = 1
data = np.repeat(GroundTruth_data,n_batch,axis=0)
plot_simulations(data, 0.68, 0.65, 101, n_batch, 2) #plot_simulations(data, p, q, max number of voters, number of batches, step (keep the number of voters odd))
plot_simulations_p(data, 15, 0.03, 30, n_batch) #plot_simulations_p(data,number of voters, q, number of values of p)

###########################################
######## Part II: Real annotations ########
###########################################

# Real annotations
compare_methods(workersA, GroundTruth, 350,25, 30) #compare_methods(annotation data, groundtruth data, size of batches, number of batches, maximum iterations)
