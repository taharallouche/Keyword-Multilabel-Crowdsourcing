"""
This file contains the functions related to the second part, namely:
-Computing the weighted MLE given votes, noise parameters and priors
-Executing an EM algorithm for the estimation of the voters' noise parameters and the groundtruth.
-Comparing the methods (MLE, MLE with uniform prior, Majority)
"""


###################################################
######Importing libraries and function files#######
###################################################

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from numpy import random as npr
from scipy import stats



def weighted_MLE(votes, param, priors):
    """
    Given the annotations of some voters and their individual noise parameters and the prior of each label, this function computes the MLE for the instance
    :param votes: A dataset of annotations
    :param param: A dataframe of individual noise parameters
    :param priors: An array containing the prior of each label
    :return: an array containing the MLE for the m labels
    """
    col = ["Basketball", "Children", "Gym class", "Graduation", "Other", "Soccer", "Birthday", "Parade", "Fireworks",
           "Cake"]
    workersA_col = col
    res = np.zeros(len(workersA_col))
    beta = -sum(
        [np.log(1 - param[param["worker_id"] == t]["p"].to_numpy()) - np.log(
            1 - param[param["worker_id"] == t]["q"].to_numpy()) for
         t in list(votes.worker_id)])
    # print(beta)
    for i in range(len(workersA_col)):
        app = param[param["worker_id"].isin(list(votes[votes[workersA_col[i]] == 1].worker_id))]["weight"].sum(
            axis=0)
        # print(app)
        if app >= beta + np.log((1 - priors[i]) / priors[i]):
            res[i] = 1
    return res


def compute_param(votes, agg, param):
    """
    Given the annotations and the groundtruth, estimate the individual noise parameters
    :param votes: A dataframe of the annotations of a set of instances by some voters
    :param agg: A dataframe containing the (estimated) groundtruth
    :param param: A dataframe of individual noise parameters
    """
    col = ["Basketball", "Children", "Gym class", "Graduation", "Other", "Soccer", "Birthday", "Parade", "Fireworks",
           "Cake"]
    workersA_col = col
    k = 0
    param["po"] = np.zeros(param.shape[0])
    param["ne"] = np.zeros(param.shape[0])
    param["total_p"] = np.zeros(param.shape[0])
    param["total_n"] = np.zeros(param.shape[0])
    for keyword in list(agg['Keyword']):
        print("ITEM ", k)
        k += 1
        for label in workersA_col:
            if agg.loc[agg["Keyword"] == keyword, label].to_numpy() == 1:
                param.loc[param["worker_id"].isin(list(votes[votes["item"] == keyword].worker_id)), "total_p"] += 1
                param.loc[
                    param["worker_id"].isin(list(votes[(votes["item"] == keyword) & (votes[label] == 1)].worker_id)),
                    "po"] += 1
            else:
                param.loc[param["worker_id"].isin(list(votes[votes["item"] == keyword].worker_id)), "total_n"] += 1
                param.loc[
                    param["worker_id"].isin(list(votes[(votes["item"] == keyword) & (votes[label] == 1)].worker_id)),
                    "ne"] += 1
    param.loc[(param["total_p"] > 0) & (param["po"] == 0), "p"] = 0.001
    param.loc[(param["total_p"] > 0) & (param["po"] == param["total_p"]), "p"] = 0.999
    param.loc[(param["total_p"] > 0) & (param["po"] < param["total_p"]) & (param["po"] > 0), "p"] = param["po"] / param[
        "total_p"]
    param.loc[(param["total_n"] > 0) & (param["ne"] == 0), "q"] = 0.001
    param.loc[(param["total_n"] > 0) & (param["ne"] == param["total_n"]), "q"] = 0.999
    param.loc[(param["total_n"] > 0) & (param["ne"] < param["total_n"]) & (param["ne"] > 0), "q"] = param["ne"] / param[
        "total_n"]
    param["weight"] = np.log(param["p"] * (1 - param["q"]) / param["q"] / (1 - param["p"]))


def compute_majority(votes):
    """
    Compute the majority outcome label by label given a set of votes
    :param votes: A dataframe of votes
    :return: An array of aggregated labels by the majority rule
    """
    items = votes.item.unique()
    n = votes.worker_id.unique().shape[0]
    col = ["Basketball", "Children", "Gym class", "Graduation", "Other", "Soccer", "Birthday", "Parade", "Fireworks",
           "Cake"]
    workersA_col = col
    agg = pd.DataFrame(columns=["Keyword"] + workersA_col)
    agg["Keyword"] = items
    k = 0
    for item in agg["Keyword"]:
        print(k)
        k += 1
        for label in workersA_col:
            agg.loc[agg["Keyword"] == item, label] = sum(votes.loc[votes["item"] == item, label]) >= 0.5 * n
    return agg


def weighted_maj(votes, param):
    """
    Given a set of votes and individual noise parameter compute the outcome of the weighted majority
    :param votes: A dataframe of votes
    :param param: A dataframe of noise parameters
    :return:
    """
    col = ["Basketball", "Children", "Gym class", "Graduation", "Other", "Soccer", "Birthday", "Parade", "Fireworks",
           "Cake"]
    workersA_col = col
    res = np.zeros(len(workersA_col))
    beta = -0.5 * sum(
        [np.log(1 - param[param["worker_id"] == t]["p"].to_numpy()) - np.log(
            param[param["worker_id"] == t]["p"].to_numpy()) for
         t in list(votes.worker_id)])
    # print(beta)
    for i in range(len(workersA_col)):
        app = param[param["worker_id"].isin(list(votes[votes[workersA_col[i]] == 1].worker_id))]["weight"].sum(
            axis=0)
        # print(app)
        if app >= beta:
            res[i] = 1
    return res


def compute_param_maj(votes, agg, param):
    """
    Given a set of votes and a (estimated) groundtruth update the noise parameters.
    :param votes: A dataframe of annotations
    :param agg: A (estimated) groundtruth
    :param param: A dataframe of individual noise parameters
    """
    col = ["Basketball", "Children", "Gym class", "Graduation", "Other", "Soccer", "Birthday", "Parade", "Fireworks",
           "Cake"]
    workersA_col = col
    k = 0
    param["correct"] = np.zeros(param.shape[0])
    param["total"] = np.zeros(param.shape[0])
    for keyword in list(agg['Keyword']):
        print("ITEM ", k)
        k += 1
        for label in workersA_col:
            if agg.loc[agg["Keyword"] == keyword, label].to_numpy() == 1:
                param.loc[param["worker_id"].isin(list(votes[votes["item"] == keyword].worker_id)), "total"] += 1
                param.loc[
                    param["worker_id"].isin(list(votes[(votes["item"] == keyword) & (votes[label] == 1)].worker_id)),
                    "correct"] += 1
            else:
                param.loc[param["worker_id"].isin(list(votes[votes["item"] == keyword].worker_id)), "total"] += 1
                param.loc[
                    param["worker_id"].isin(list(votes[(votes["item"] == keyword) & (votes[label] == 0)].worker_id)),
                    "correct"] += 1
    param.loc[(param["total"] > 0) & (param["correct"] == 0), "p"] = 0.001
    param.loc[(param["total"] > 0) & (param["correct"] == param["total"]), "p"] = 0.999
    param.loc[(param["total"] > 0) & (param["correct"] < param["total"]) & (param["p"] > 0), "p"] = param["correct"] / \
                                                                                                    param[
                                                                                                        "total"]
    param["weight"] = np.log(param["p"] / (1 - param["p"]))


def expect_max_maj(votes, epsi, n_iter, p):
    """
    Given a dataframe of votes, execute the EM algorithm for the weighted majority method
    :param votes: A dataframe of votes
    :param epsi: epsilon tolerance
    :param n_iter: maximum number of iterations
    :param p: overall noise parameters
    :return: estimated groundtruth + estimated noise parameters
    """
    workers_id = votes.worker_id.unique()
    param_data = {"worker_id": workers_id, "p": p * np.ones(workers_id.shape)}
    param = pd.DataFrame(param_data)
    param["weight"] = np.log(param["p"] / (1 - param["p"]))
    # votes = pd.merge(annotations, param, on="worker_id")
    param["correct"] = np.zeros(param.shape[0])
    param["total"] = np.zeros(param.shape[0])
    items = votes.item.unique()
    workersA_col = ["Basketball", "Children", "Gym class", "Graduation", "Other",
                    "Soccer", "Birthday", "Beach", "Parade", "Guitar", "Rain", "Fireworks",
                    "Floating lanterns", "Cake", "Sleeping", "Birthday drinks", "Boat"]
    col = ["Basketball", "Children", "Gym class", "Graduation", "Other", "Soccer", "Birthday", "Parade", "Fireworks",
           "Cake"]
    workersA_col = col
    agg = pd.DataFrame(columns=["Keyword"] + workersA_col)
    agg["Keyword"] = items
    k = 0
    for item in items:
        print("Item: ", k)
        k += 1
        agg.loc[agg["Keyword"] == item, workersA_col] = weighted_maj(votes[votes["item"] == item], param)
    param1 = param.copy()
    # for i in range(len(priors)):
    #   priors[i] = max([sum(agg.loc[:, workersA_col[i]]) / agg.shape[0], 0.001])
    #  print("prior: ", priors[i])
    iter = 0
    test = True
    error = 0
    while (iter == 0) or (test and
                          (iter <= n_iter) and (
                                  np.max(abs(param[["p"]].to_numpy() - param1[["p"]].to_numpy())) > epsi)):
        iter += 1
        param1 = param[["p"]].copy()
        print("######### ", iter, " #########")
        k = 0
        for item in items:
            k += 1
            agg.loc[agg["Keyword"] == item, workersA_col] = weighted_maj(votes.loc[votes.item == item], param)
            print("Item:  ", k)
        k = 0
        # for i in range(len(priors)):
        #   priors[i] = max([sum(agg.loc[:, workersA_col[i]]) / agg.shape[0], 0.001])
        compute_param_maj(votes, agg, param)
        print("Iter, ", iter, " FINISHED ")
    return agg, param[["p"]]


def expect_max_prior(votes, epsi, n_iter, prior, p, q):
    """
    Given a set of annotations, execute the EM algorithm to estimate the groundtruth, the priors and the noise parameters via the MLE
    :param votes: A dataframe of annotations
    :param epsi: epsilon tolerance
    :param n_iter: maximum number of iterations
    :param prior: initial priors
    :param p: initital noise parameters
    :param q: initial noise parameters
    :return: estimated groundtruth + estimated noise parameters
    """
    workers_id = votes.worker_id.unique()
    param_data = {"worker_id": workers_id, "p": p * np.ones(workers_id.shape), 'q': q * np.ones(workers_id.shape)}
    param = pd.DataFrame(param_data)
    param["weight"] = np.log(param["p"] * (1 - param["q"]) / param["q"] / (1 - param["p"]))
    # votes = pd.merge(annotations, param, on="worker_id")
    param["po"] = np.zeros(param.shape[0])
    param["ne"] = np.zeros(param.shape[0])
    param["total_p"] = np.zeros(param.shape[0])
    param["total_n"] = np.zeros(param.shape[0])
    items = votes.item.unique()
    workersA_col = ["Basketball", "Children", "Gym class", "Graduation", "Other",
                    "Soccer", "Birthday", "Beach", "Parade", "Guitar", "Rain", "Fireworks",
                    "Floating lanterns", "Cake", "Sleeping", "Birthday drinks", "Boat"]
    col = ["Basketball", "Children", "Gym class", "Graduation", "Other", "Soccer", "Birthday", "Parade", "Fireworks",
           "Cake"]
    workersA_col = col
    agg = pd.DataFrame(columns=["Keyword"] + workersA_col)
    agg["Keyword"] = items
    k = 0
    priors = prior
    for item in items:
        print("Item: ", k)
        k += 1
        agg.loc[agg["Keyword"] == item, workersA_col] = weighted_MLE(votes[votes["item"] == item], param, priors)
    param1 = param.copy()
    for i in range(len(priors)):
        priors[i] = max([sum(agg.loc[:, workersA_col[i]]) / agg.shape[0], 0.001])
        print("prior: ", priors[i])
    iter = 0
    test = True
    error = 0
    while (iter == 0) or (test and
                          (iter <= n_iter) and (
                                  np.max(abs(param[["p", "q"]].to_numpy() - param1[["p", "q"]].to_numpy())) > epsi)):
        iter += 1
        param1 = param[["p", "q"]].copy()
        print("######### ", iter, " #########")
        k = 0
        for item in items:
            k += 1
            agg.loc[agg["Keyword"] == item, workersA_col] = weighted_MLE(votes.loc[votes.item == item], param, priors)
            print("Item:  ", k)
        k = 0
        for i in range(len(priors)):
            priors[i] = max([sum(agg.loc[:, workersA_col[i]]) / agg.shape[0], 0.001])
        compute_param(votes, agg, param)
        print("Iter, ", iter, " FINISHED ")
    return agg, param


# Recuperate estimated noise parameters and generate votes accordingly
def generated_votes(GroundData, votes, param):
    """
    Estimate groundtruth via MLE when real annotations for each voter are substituted by votes generated according to noise model with estimated noise parameters.
    :param GroundData: groundtruth data
    :param votes: a dataframe of votes
    :param param: estimated individual noise parameters
    :return: estimated groundtruth
    """
    col = ["Basketball", "Children", "Gym class", "Graduation", "Other", "Soccer", "Birthday", "Parade", "Fireworks",
           "Cake"]
    workersA_col = col
    # Generate votes according to param
    for i in range(votes.shape[0]):
        for label in col:
            item = votes.iloc[i, :]["item"]
            worker_id = votes.iloc[i, :]["worker_id"]
            if GroundData[GroundData["Keyword"] == item][label].to_numpy() == 1:
                r = npr.random_sample(1)
                if r <= param[param["worker_id"] == votes.iloc[i, :]["worker_id"]]["p"].to_numpy():
                    votes.iloc[i,votes.columns.get_loc(label)]=1
                else:
                    votes.iloc[i, votes.columns.get_loc(label)] = 0
            else:
                r = npr.random_sample(1)
                if r <= param[param["worker_id"] == votes.iloc[i, :]["worker_id"]]["q"].to_numpy():
                    votes.iloc[i, votes.columns.get_loc(label)] = 1
                else:
                    votes.iloc[i, votes.columns.get_loc(label)] = 0
    # Compute real priors
    priors = np.zeros(len(workersA_col))
    for i in range(len(priors)):
        priors[i] = max([sum(GroundData.loc[:, workersA_col[i]]) / GroundData.shape[0], 0.0001])
    # Aggregate
    items = votes.item.unique()
    agg = pd.DataFrame(columns=["Keyword"] + workersA_col)
    agg["Keyword"] = items
    k = 0
    for item in items:
        print("Item: ", k)
        k += 1
        agg.loc[agg["Keyword"] == item, workersA_col] = weighted_MLE(votes[votes["item"] == item], param, priors)
    return agg


def expect_max(votes, epsi, n_iter, p, q):
    """
    Given a set of annotations, execute the EM algorithm to estimate the groundtruth, the priors and the noise parameters via the MLE with the assumption that the priors are uniform
    :param votes: A dataframe of annotations
    :param epsi: epsilon tolerance
    :param n_iter: maximum number of iterations
    :param p: initital noise parameters
    :param q: initial noise parameters
    :return: estimated groundtruth + estimated noise parameters
    """
    workers_id = votes.worker_id.unique()
    param_data = {"worker_id": workers_id, "p": p * np.ones(workers_id.shape), 'q': q * np.ones(workers_id.shape)}
    param = pd.DataFrame(param_data)
    param["weight"] = np.log(param["p"] * (1 - param["q"]) / param["q"] / (1 - param["p"]))
    # votes = pd.merge(annotations, param, on="worker_id")
    param["po"] = np.zeros(param.shape[0])
    param["ne"] = np.zeros(param.shape[0])
    param["total_p"] = np.zeros(param.shape[0])
    param["total_n"] = np.zeros(param.shape[0])
    items = votes.item.unique()
    col = ["Basketball", "Children", "Gym class", "Graduation", "Other", "Soccer", "Birthday", "Parade", "Fireworks",
           "Cake"]
    workersA_col = col
    agg = pd.DataFrame(columns=["Keyword"] + workersA_col)
    agg["Keyword"] = items
    k = 0
    priors = 0.5 * np.ones(len(workersA_col))
    for item in items:
        print("Item: ", k)
        k += 1
        agg.loc[agg["Keyword"] == item, workersA_col] = weighted_MLE(votes[votes["item"] == item], param, priors)
    param1 = param.copy()
    iter = 0
    test = True
    error = 0
    while (iter == 0) or (test and
                          (iter <= n_iter) and (
                                  np.max(abs(param[["p", "q"]].to_numpy() - param1[["p", "q"]].to_numpy())) > epsi)):
        iter += 1
        param1 = param[["p", "q"]].copy()
        print("######### ", iter, " #########")
        k = 0
        for item in items:
            k += 1
            agg.loc[agg["Keyword"] == item, workersA_col] = weighted_MLE(votes.loc[votes.item == item], param, priors)
            print("Item:  ", k)
        k = 0
        compute_param(votes, agg, param)
        print("Iter, ", iter, " FINISHED ")
    return agg, param[["p", "q"]]


def compare_methods(annotations, GroundTruth, Ground_size, n_batch, iter_max):
    col = ["Basketball", "Children", "Gym class", "Graduation", "Other", "Soccer", "Birthday", "Parade", "Fireworks",
           "Cake"]
    workersA_col = col
    Loss = np.zeros([2, 5])  # Hamming 0/1
    Loss_list = np.zeros([2, 5, n_batch])  # 3 losses #4 rules #n_batch losses
    p_hist = []
    q_hist = []
    ov_prec = []
    Interval = np.zeros([3, 4])
    for batch in range(n_batch):
        print("######### Batch:", batch)
        prior = np.random.uniform(0.01, 0.5, len(workersA_col))
        p = random.uniform(0.401, 0.899)
        q = random.uniform(0.01, 0.399)
        GroundData = GroundTruth.iloc[random.sample(range(0, GroundTruth.shape[0]), Ground_size), :]
        GroundData = GroundData.drop_duplicates(subset=["Keyword"], keep='first')
        # MLE with prior
        agg_mle, param = expect_max_prior(annotations.loc[annotations["item"].isin(GroundData["Keyword"])], 0.01,
                                          iter_max, prior, p, q)
        p_hist += list(param["p"].to_numpy())
        q_hist += list(param["q"].to_numpy())
        l_01 = sum(np.array_equal(agg_mle.loc[agg_mle["Keyword"] == item, workersA_col].to_numpy(),
                                  GroundData.loc[GroundData["Keyword"] == item, workersA_col].to_numpy()) for item in
                   agg_mle["Keyword"])
        Loss[1, 0] += l_01 / Ground_size
        Loss_list[1, 0, batch] = l_01 / Ground_size
        l_ham = sum(np.array_equal(agg_mle.loc[agg_mle["Keyword"] == item, label].to_numpy(),
                                   GroundData.loc[GroundData["Keyword"] == item, label].to_numpy()) for item in
                    agg_mle["Keyword"] for label in workersA_col)
        Loss[0, 0] += l_ham / Ground_size / len(workersA_col)
        Loss_list[0, 0, batch] = l_ham / Ground_size / len(workersA_col)
        # Generated votes
        agg_gen = generated_votes(GroundData, annotations.loc[annotations["item"].isin(GroundData["Keyword"])], param)
        l_01 = sum(np.array_equal(agg_gen.loc[agg_gen["Keyword"] == item, workersA_col].to_numpy(),
                                  GroundData.loc[GroundData["Keyword"] == item, workersA_col].to_numpy()) for item in
                   agg_gen["Keyword"])
        Loss[1, 1] += l_01 / Ground_size
        Loss_list[1, 1, batch] = l_01 / Ground_size
        l_ham = sum(np.array_equal(agg_gen.loc[agg_mle["Keyword"] == item, label].to_numpy(),
                                   GroundData.loc[GroundData["Keyword"] == item, label].to_numpy()) for item in
                    agg_gen["Keyword"] for label in workersA_col)
        Loss[0, 1] += l_ham / Ground_size / len(workersA_col)
        Loss_list[0, 1, batch] = l_ham / Ground_size / len(workersA_col)

        # MLE with uniform prior
        agg_mle_u, param = expect_max(annotations.loc[annotations["item"].isin(GroundData["Keyword"])], 0.01,
                                      iter_max, p, q)
        l_01 = sum(np.array_equal(agg_mle_u.loc[agg_mle_u["Keyword"] == item, workersA_col].to_numpy(),
                                  GroundData.loc[GroundData["Keyword"] == item, workersA_col].to_numpy()) for item in
                   agg_mle_u["Keyword"])
        Loss[1, 2] += l_01 / Ground_size
        Loss_list[1, 2, batch] = l_01 / Ground_size
        l_ham = sum(np.array_equal(agg_mle_u.loc[agg_mle_u["Keyword"] == item, label].to_numpy(),
                                    GroundData.loc[GroundData["Keyword"] == item, label].to_numpy()) for item in
                     agg_mle_u["Keyword"] for label in workersA_col)
        Loss[0, 2] += l_ham / Ground_size / len(workersA_col)
        Loss_list[0, 2, batch] = l_ham / Ground_size / len(workersA_col)

        # Majority
        agg_smr= compute_majority(annotations.loc[annotations["item"].isin(GroundData["Keyword"])])
        l_01 = sum(np.array_equal(agg_smr.loc[agg_smr["Keyword"] == item, workersA_col].to_numpy(),
                                  GroundData.loc[GroundData["Keyword"] == item, workersA_col].to_numpy()) for item in
                   agg_smr["Keyword"])
        Loss[1, 3] += l_01 / Ground_size
        Loss_list[1, 3, batch] = l_01 / Ground_size
        l_ham = sum(np.array_equal(agg_smr.loc[agg_smr["Keyword"] == item, label].to_numpy(),
                                   GroundData.loc[GroundData["Keyword"] == item, label].to_numpy()) for item in
                    agg_smr["Keyword"] for label in workersA_col)
        Loss[0, 3] += l_ham / Ground_size / len(workersA_col)
        Loss_list[0, 3, batch] = l_ham / Ground_size / len(workersA_col)

        # Weighted Majority
        p = random.uniform(0.6, 0.95)
        agg_maj, param = expect_max_maj(annotations.loc[annotations["item"].isin(GroundData["Keyword"])], 0.01,
                                       iter_max, p)
        ov_prec += list(param["p"].to_numpy())
        l_01 = sum(np.array_equal(agg_maj.loc[agg_maj["Keyword"] == item, workersA_col].to_numpy(),
                                  GroundData.loc[GroundData["Keyword"] == item, workersA_col].to_numpy()) for item in
                   agg_maj["Keyword"])
        Loss[1, 4] += l_01 / Ground_size
        Loss_list[1, 4, batch] = l_01 / Ground_size
        l_ham = sum(np.array_equal(agg_maj.loc[agg_maj["Keyword"] == item, label].to_numpy(),
                                   GroundData.loc[GroundData["Keyword"] == item, label].to_numpy()) for item in
                    agg_maj["Keyword"] for label in workersA_col)
        Loss[0, 4] += l_ham / Ground_size / len(workersA_col)
        Loss_list[0, 4, batch] = l_ham / Ground_size / len(workersA_col)

    Loss = (1 / n_batch) * Loss

    for i in range(2):
        for j in range(5):
            Interval[i, j] = 1.96 * np.sqrt((Loss[i, j] * (1 - Loss[i, j])) / n_batch)
    print("###### Loss + 0.95 IC #####")
    print("Hamming MLE_prior: ", Loss[0, 0], "+-", Interval[0, 0])
    print("0/1 MLE_prior: ", Loss[1, 0], "+-", Interval[1, 0])
    print("##########")
    print("Hamming Generated: ", Loss[0, 1], "+-", Interval[0, 1])
    print("0/1 Generated: ", Loss[1, 1], "+-", Interval[1, 1])
    print("##########")
    print("Hamming MLE_uniform: ", Loss[0, 2], "+-", Interval[0, 2])
    print("0/1 Generated: ", Loss[1, 2], "+-", Interval[1, 2])
    print("##########")
    print("Hamming Simple Majority: ", Loss[0, 3], "+-", Interval[0, 3])
    print("0/1 Simple Majority: ", Loss[1, 3], "+-", Interval[1, 3])
    print("##########")
    print("Hamming Weighted Majority: ", Loss[0, 4], "+-", Interval[0, 4])
    print("0/1 Weighted Majority: ", Loss[1, 4], "+-", Interval[1, 4])
    print("######## Test de Student: MLE vs MLE_uniform #########")
    t, p = scipy.stats.ttest_rel(Loss_list[0, 0, :], Loss_list[0, 1, :])
    print("Student test for Hamming Loss MLE/MLE_u: t= ", t, " ,p= ", p)
    t, p = scipy.stats.ttest_rel(Loss_list[1, 0, :], Loss_list[1, 1, :])
    print("Student test for 0/1 Loss MLE/MLE_u: t= ", t, " ,p= ", p)
    print("######## Test de Wilcoxon: MLE vs MLE_uniform #########")
    t, p = scipy.stats.wilcoxon(Loss_list[0, 0, :], Loss_list[0, 1, :])
    print("Wilcoxon test for Hamming Loss MLE/MLE_u: t= ", t, " ,p= ", p)
    t, p = scipy.stats.wilcoxon(Loss_list[1, 0, :], Loss_list[1, 1, :])
    print("Wilcoxon test for 0/1 Loss MLE/MLE_u: t= ", t, " ,p= ", p)
    print("######## Test de Student MLE vs Weighted Majority #########")
    t, p = scipy.stats.ttest_rel(Loss_list[0, 0, :], Loss_list[0, 4, :])
    print("Student test for Hamming Loss MLE/Weighted Maj: t= ", t, " ,p= ", p)
    t, p = scipy.stats.ttest_rel(Loss_list[1, 4, :], Loss_list[1, 4, :])
    print("Student test for 0/1 Loss MLE/Weighted Maj: t= ", t, " ,p= ", p)
    print("######## Test de Wilcoxon #########")
    t, p = scipy.stats.wilcoxon(Loss_list[0, 0, :], Loss_list[0, 1, :])
    print("Wilcoxon test for Hamming Loss MLE/Weighted Maj: t= ", t, " ,p= ", p)
    t, p = scipy.stats.wilcoxon(Loss_list[1, 0, :], Loss_list[1, 1, :])
    print("Wilcoxon test for 0/1 Loss MLE/Weighted Maj: t= ", t, " ,p= ", p)

    fig = plt.figure()
    bins = np.linspace(0, 1, 40)
    plt.hist(np.array(p_hist), bins, weights=np.ones(len(p_hist)) / len(p_hist), label='p', alpha=0.5, hatch="/",fill=True)
    plt.hist(np.array(q_hist), bins, weights=np.ones(len(q_hist)) / len(q_hist), label='q', alpha=0.5,hatch="|",fill=True)
    plt.hist(np.array(ov_prec), bins, weights=np.ones(len(q_hist)) / len(q_hist), label='overall', alpha=0.5)
    plt.legend(loc='best')
    plt.title("Histogram of precision parameters' distributions")
    plt.show()
    plt.savefig("Histogram of p, q and overall precision.png")
