"""
This file contains all the functions necessary for the simulation part, namely:
-Generating votes according to the noise model for some noise parameters.
-Apply different aggregation rules.
-Compute and plot the Hamming and 0/1 Losses
"""

###################################################
######Importing libraries and function files#######
###################################################
import numpy as np
from numpy import random as npr
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from math import comb
from numpy.core._multiarray_umath import ndarray


# Generate a single vote
def gen_vote(inst, p, q):
    """
    Generate a single vote according to noise model with parameters (p,q) and m groundtruth labels inst
    :param inst: an array of binary labels constituting the groundtruth
    :param p: first noise parameter: True Positive probability
    :param q: second noise parameter: False Positive probability
    :return: an array of binary entries describing the approval ballot generated
    """
    m = len(inst)
    vote = np.copy(inst)
    for i in range(m):
        if inst[i] == 0:
            # npr.seed()
            r = npr.random_sample(1)
            if r <= q:
                vote[i] = 1
        else:
            # npr.seed()
            r = npr.random_sample(1)
            if r <= 1 - p:
                vote[i] = 0
    return vote


# Generate n votes
def gen_votes(inst, p, q, n):
    """
    Generate n votes according to the noise model with parameters (p,q) given groundtruth labels inst
    :param inst: an array of binary labels constituting the groundtruth
    :param p: first noise parameter: True Positive probability
    :param q: q: second noise parameter: False Positive probability
    :param n: number of voters
    :return: an array (n x m) of n approval ballots.
    """
    m = len(inst)
    votes = np.zeros([n, m])
    for i in range(n):
        votes[i, :] = gen_vote(inst, p, q)
    return votes


# Compute MLE with uniform prior from votes on m labels
def compute_mle(votes, p, q):
    """
    Compute the MLE assuming the prior is uniform, given a set of votes and the noise parameters from which the votes are generated.
    :param votes: an array (n x m) of n approval ballots (m sized binary line)
    :param p: first noise parameter: True Positive probability
    :param q: second noise parameter: False Positive probability
    :return: an array of m binary labels
    """
    n = votes.shape[0]
    m = votes.shape[1]
    agg = np.zeros([m])
    alpha = (np.log((1 - q) / (1 - p))) / (np.log((p * (1 - q)) / (q * (1 - p))))
    threshold = alpha * n
    for i in range(m):
        if sum(votes[:, i]) > threshold:
            agg[i] = 1
        elif sum(votes[:, i]) == threshold:
            agg[i] = (npr.random_sample(1) >= 0.5)
    return agg


# Compute Majority from votes
def compute_maj(votes):
    """
    Compute the majority rule outcome label by label given a set of approval ballots.
    :param votes: an array (n x m) of n approval ballots (m sized binary line)
    :return: an array of m binary labels
    """
    n = votes.shape[0]
    m = votes.shape[1]
    agg = np.zeros([m])
    threshold = 0.5 * n
    for i in range(m):
        if sum(votes[:, i]) > threshold:
            agg[i] = 1
        elif sum(votes[:, i]) == threshold:
            agg[i] = (npr.random_sample(1) >= 0.5)
    return agg


# Compute Modal from votes
def compute_mode(votes):
    """
    Compute the rule rule outcome given a set of approval ballots.
    :param votes: an array (n x m) of n approval ballots (m sized binary line)
    :return: an array of m binary labels
    """
    a = np.ascontiguousarray(votes)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _, ids, count = np.unique(a.view(void_dt).ravel(), return_index=1, return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row


# Compute MLE from votes
def compute_prior_mle(votes, p, q, gamma):
    """
    Compute the MLE assuming the prior is uniform, given a set of votes and the noise parameters from which the votes are generated.
    :param votes: an array (n x m) of n approval ballots (m sized binary line)
    :param p: first noise parameter: True Positive probability
    :param q: second noise parameter: False Positive probability
    :return: an array of m binary labels
    """
    n = votes.shape[0]
    m = votes.shape[1]
    agg = np.zeros([m])
    alpha = (np.log((1 - q) / (1 - p))) / (np.log((p * (1 - q)) / (q * (1 - p))))
    threshold = alpha * n
    for i in range(m):
        if sum(votes[:, i]) > threshold + gamma[i]:
            agg[i] = 1
        elif sum(votes[:, i]) == threshold + gamma[i]:
            agg[i] = (npr.random_sample(1) >= 0.5)
    return agg


# Run a batch of simulations on a dataset
def sim_batch(data, p, q, n_max):
    """
    Given a set of instances (binary m sized row each) data, generate votes from noise model (p,q) for each instance and for each odd number of voters less than n_max and compute the outcome according to each rule
    :param data: Dataset of groundtruth labels for a set of instances
    :param p: first noise parameter: True Positive probability
    :param q: second noise parameter: False Positive probability
    :param n_max: maximum number of voters
    :return: a 3-dim array (2 x n_max x 4) loss containing the Hamming and 0/1 loss for each number of voters for each aggregation rule
    """
    priors = np.zeros(data.shape[1])
    gamma = np.zeros(data.shape[1])
    for i in range(len(priors)):
        priors[i] = sum(data[:, i]) / data.shape[0]
        gamma[i] = (np.log((1 - priors[i]) / priors[i])) / (np.log((p * (1 - q)) / (q * (1 - p))))
    m = data.shape[1]
    loss = np.zeros([2, n_max, 4])  # 2 losses, n_max experiments, 3 rules
    results = np.zeros(
        [4, data.shape[0], m])  # 3 rules, data.shape instances, will be overwritten after each experiment
    for i in range(0, n_max, 2):
        print(i, " \n")
        for j in range(data.shape[0]):
            inst = data[j, :]
            votes = gen_votes(inst, p, q, i + 1)
            results[0, j, :] = compute_mle(votes, p, q)
            results[1, j, :] = compute_maj(votes)
            results[2, j, :] = compute_mode(votes)
            results[3, j, :] = compute_prior_mle(votes, p, q, gamma)
        loss[0, i, 0] = hamming_loss(data, results[0, :, :])
        loss[0, i, 1] = hamming_loss(data, results[1, :, :])
        loss[0, i, 2] = hamming_loss(data, results[2, :, :])
        loss[0, i, 3] = hamming_loss(data, results[3, :, :])
        loss[1, i, 0] = zero_one_loss(data, results[0, :, :])
        loss[1, i, 1] = zero_one_loss(data, results[1, :, :])
        loss[1, i, 2] = zero_one_loss(data, results[2, :, :])
        loss[1, i, 3] = zero_one_loss(data, results[3, :, :])
    return loss


def sim_p(data, n, q, nm=100):
    """
    Given a set of instances (binary m sized row each) data, generate n votes from noise model (p,q) for each instance and for the nm values p between q and 1 and compute the outcome according to each rule
    :param data: Dataset of groundtruth labels for a set of instances
    :param n: The number of voters
    :param q: second noise parameter: False Positive probability
    :param nm: number of values of p (between p and 1) to consider
    :return: a 3-dim array (2 x l x 4) loss containing the Hamming and 0/1 loss for each value of p for each aggregation rule
    """
    priors = np.zeros(data.shape[1])
    gamma = np.zeros(data.shape[1])
    for i in range(len(priors)):
        priors[i] = sum(data[:, i]) / data.shape[0]
    m = data.shape[1]
    p = np.linspace(q + 0.05, 0.95, nm)
    loss = np.zeros(
        [2, len(np.linspace(q + 0.05, 0.95, nm)), 4])  # 2 losses, number of values experiments, 3 rules
    results = np.zeros(
        [4, data.shape[0], m])  # 3 rules, data.shape instances, will be overwritten after each experiment
    for i in range(len(p) - 1):
        for k in range(m):
            gamma[k] = (np.log((1 - priors[k]) / priors[k])) / (np.log((p[i] * (1 - q)) / (q * (1 - p[i]))))
        print(i, " \n")
        for j in range(data.shape[0]):
            inst = data[j, :]
            votes = gen_votes(inst, p[i], q, n)
            results[0, j, :] = compute_mle(votes, p[i], q)
            results[1, j, :] = compute_maj(votes)
            results[2, j, :] = compute_mode(votes)
            results[3, j, :] = compute_prior_mle(votes, p[i], q, gamma)
        loss[0, i, 0] = hamming_loss(data, results[0, :, :])
        loss[0, i, 1] = hamming_loss(data, results[1, :, :])
        loss[0, i, 2] = hamming_loss(data, results[2, :, :])
        loss[0, i, 3] = hamming_loss(data, results[3, :, :])
        loss[1, i, 0] = zero_one_loss(data, results[0, :, :])
        loss[1, i, 1] = zero_one_loss(data, results[1, :, :])
        loss[1, i, 2] = zero_one_loss(data, results[2, :, :])
        loss[1, i, 3] = zero_one_loss(data, results[3, :, :])
    return loss


def simulations_p(data, q, n, nm, n_batch):
    """
    Repeat and average over n_batch the function sim_p
    :param data: Dataset of groundtruth labels for a set of instances
    :param q: second noise parameter: False Positive probability
    :param n: The number of voters
    :param nm: number of values of p (between p and 1) to consider
    :param n_batch: number of batches
    :return: a 3-dim array (2 x l x 4) loss containing the Hamming and 0/1 loss for each value of p for each aggregation rule
    """
    loss = np.zeros([2, len(np.linspace(q + 0.05, 0.95, nm)), 3])
    for i in range(n_batch):
        loss += sim_p(data, n, q, nm)
    return loss / n_batch


def simulations(data, p, q, n_max, n_batch):
    """
    repeat and average over n_batch the function sim_batch
    :param data: Dataset of groundtruth labels for a set of instances
    :param p: first noise parameter: True Positive probability
    :param q: second noise parameter: False Positive probability
    :param n_max: maximum number of voters
    :param n_batch: number of batches
    :return: a 3-dim array (2 x n_max x 4) loss containing the Hamming and 0/1 loss for each number of voters for each aggregation rule
    """
    loss = np.zeros([2, n_max, 3])
    for i in range(n_batch):
        loss += sim_batch(data, p, q, n_max)
    return loss / n_batch


def plot_simulations(data, p, q, n_max, n_batch, l=2):
    """
    Plot the Hamming and 0/1 loss with confidence intervals for the 4 aggregation rules for different number of voters whose approval ballots are generated according to the (p,q) noise model
    :param data: Dataset of groundtruth labels for a set of instances
    :param p: first noise parameter: True Positive probability
    :param q: second noise parameter: False Positive probability
    :param n_max: maximum number of voters
    :param n_batch: number of batches
    :param l: step
    """
    m = data.shape[1]
    # loss = simulations(data, p, q, n_max, n_batch)
    loss = sim_batch(data, p, q, n_max)
    # Hamming Loss
    fig = plt.figure()
    plt.errorbar(range(1, n_max + 1, l), loss[0, 0:n_max:l, 0], label='MLE_uniform', linestyle="dashed")
    plt.fill_between(range(1, n_max + 1, l), loss[0, 0:n_max:l, 0] - 1.96 * np.sqrt(
        (loss[0, 0:n_max:l, 0] * (1 - loss[0, 0:n_max:l, 0])) / data.shape[0] / m),
                     loss[0, 0:n_max:l, 0] + 1.96 * np.sqrt(
                         (loss[0, 0:n_max:l, 0] * (1 - loss[0, 0:n_max:l, 0])) / data.shape[0] / m), color='b',
                     alpha=0.1)
    plt.errorbar(range(1, n_max + 1, l), loss[0, 0:n_max:l, 1], label='Majority', linestyle="dashdot")
    plt.fill_between(range(1, n_max + 1, l), loss[0, 0:n_max:l, 1] - 1.96 * np.sqrt(
        (loss[0, 0:n_max:l, 1] * (1 - loss[0, 0:n_max:l, 1])) / data.shape[0] / m),
                     loss[0, 0:n_max:l, 1] + 1.96 * np.sqrt(
                         (loss[0, 0:n_max:l, 1] * (1 - loss[0, 0:n_max:l, 1])) / data.shape[0] / m), color='orange',
                     alpha=0.1)
    plt.errorbar(range(1, n_max + 1, l), loss[0, 0:n_max:l, 2], label='Modal', linestyle="dotted")
    plt.fill_between(range(1, n_max + 1, l), loss[0, 0:n_max:l, 2] - 1.96 * np.sqrt(
        (loss[0, 0:n_max:l, 2] * (1 - loss[0, 0:n_max:l, 2])) / data.shape[0] / m),
                     loss[0, 0:n_max:l, 2] + 1.96 * np.sqrt(
                         (loss[0, 0:n_max:l, 2] * (1 - loss[0, 0:n_max:l, 2])) / data.shape[0] / m), color='green',
                     alpha=0.1)
    plt.errorbar(range(1, n_max + 1, l), loss[0, 0:n_max:l, 3], label='MLE', linestyle="solid")
    plt.fill_between(range(1, n_max + 1, l), loss[0, 0:n_max:l, 3] - 1.96 * np.sqrt(
        (loss[0, 0:n_max:l, 3] * (1 - loss[0, 0:n_max:l, 3])) / data.shape[0] / m),
                     loss[0, 0:n_max:l, 3] + 1.96 * np.sqrt(
                         (loss[0, 0:n_max:l, 3] * (1 - loss[0, 0:n_max:l, 3])) / data.shape[0] / m), color='red',
                     alpha=0.1)
    plt.legend(loc='upper right')
    plt.title("Hamming loss, (p,q)=(" + str(p) + "," + str(q) + ")")
    plt.xlabel("Number of voters")
    plt.xticks(np.arange(1, n_max + 1, 10), np.arange(1, n_max + 1, 10))
    plt.ylabel("Hamming loss")
    title = "Hamming_n" + str(n_max) + "_b" + str(n_batch) + "_p" + str(p) + "_q" + str(q) + ".png"
    plt.savefig(title)

    # 0-1 Loss
    fig1 = plt.figure()
    plt.errorbar(range(1, n_max + 1, l), loss[1, 0:n_max:l, 0], label='MLE_uniform', linestyle="dashed")
    plt.errorbar(range(1, n_max + 1, l), loss[1, 0:n_max:l, 1], label='Majority', linestyle="dashdot")
    plt.errorbar(range(1, n_max + 1, l), loss[1, 0:n_max:l, 2], label='Modal', linestyle="dotted")
    plt.fill_between(range(1, n_max + 1, l), loss[1, 0:n_max:l, 0] - 1.96 * np.sqrt(
        (loss[1, 0:n_max:l, 0] * (1 - loss[1, 0:n_max:l, 0])) / data.shape[0]),
                     loss[1, 0:n_max:l, 0] + 1.96 * np.sqrt(
                         (loss[1, 0:n_max:l, 0] * (1 - loss[1, 0:n_max:l, 0])) / data.shape[0]), color='b',
                     alpha=0.1)
    plt.fill_between(range(1, n_max + 1, l), loss[1, 0:n_max:l, 1] - 1.96 * np.sqrt(
        (loss[1, 0:n_max:l, 1] * (1 - loss[1, 0:n_max:l, 1])) / data.shape[0]),
                     loss[1, 0:n_max:l, 1] + 1.96 * np.sqrt(
                         (loss[1, 0:n_max:l, 1] * (1 - loss[1, 0:n_max:l, 1])) / data.shape[0]), color='orange',
                     alpha=0.1)
    plt.fill_between(range(1, n_max + 1, l), loss[1, 0:n_max:l, 2] - 1.96 * np.sqrt(
        (loss[1, 0:n_max:l, 2] * (1 - loss[1, 0:n_max:l, 2])) / data.shape[0]),
                     loss[1, 0:n_max:l, 2] + 1.96 * np.sqrt(
                         (loss[1, 0:n_max:l, 2] * (1 - loss[1, 0:n_max:l, 2])) / data.shape[0]), color='green',
                     alpha=0.1)
    plt.errorbar(range(1, n_max + 1, l), loss[1, 0:n_max:l, 3], label='MLE', linestyle="solid")
    plt.fill_between(range(1, n_max + 1, l), loss[1, 0:n_max:l, 3] - 1.96 * np.sqrt(
        (loss[1, 0:n_max:l, 3] * (1 - loss[1, 0:n_max:l, 3])) / data.shape[0] / m),
                     loss[1, 0:n_max:l, 3] + 1.96 * np.sqrt(
                         (loss[1, 0:n_max:l, 3] * (1 - loss[1, 0:n_max:l, 3])) / data.shape[0] / m), color='red',
                     alpha=0.1)
    plt.legend(loc='upper right')
    plt.title("0-1 loss, (p,q)=(" + str(p) + "," + str(q) + ")")
    plt.xlabel("Number of voters")
    plt.xticks(np.arange(1, n_max + 1, 10), np.arange(1, n_max + 1, 10))
    plt.ylabel("0-1 loss")
    title = "ZeroOne_n" + str(n_max) + "_b" + str(n_batch) + "_p" + str(p) + "_q" + str(q) + ".png"
    plt.savefig(title)


def plot_simulations_p(data, n, q, num, n_batch):
    """
    Plot the Hamming and 0/1 loss with confidence intervals for the 4 aggregation rules for different values of p from which n approval ballots are generated according to the (p,q) noise model
    :param data: Dataset of groundtruth labels for a set of instances
    :param n: number of voters
    :param q: second noise parameter: False Positive probability
    :param num: number of values of p between q and 1
    :param n_batch: number of batches
    """
    m = data.shape[1]
    loss = sim_p(data, n, q, num)
    # Hamming Loss
    fig = plt.figure()
    plt.errorbar(np.linspace(q + 0.05, 0.95, num), loss[0, :, 0], label='MLE_uniform', linestyle="dashed")
    plt.errorbar(np.linspace(q + 0.05, 0.95, num), loss[0, :, 1], label='Majority', linestyle="dashdot")
    plt.errorbar(np.linspace(q + 0.05, 0.95, num), loss[0, :, 2], label='Modal', linestyle="dotted")
    plt.errorbar(np.linspace(q + 0.05, 0.95, num), loss[0, :, 3], label='MLE', linestyle="solid")
    plt.fill_between(np.linspace(q + 0.05, 0.95, num), loss[0, :, 0] - 1.96 * np.sqrt(
        (loss[0, :, 0] * (1 - loss[0, :, 0])) / data.shape[0] / m),
                     loss[0, :, 0] + 1.96 * np.sqrt(
                         (loss[0, :, 0] * (1 - loss[0, :, 0])) / data.shape[0] / m), color='b',
                     alpha=0.1)
    plt.fill_between(np.linspace(q + 0.05, 0.95, num), loss[0, :, 1] - 1.96 * np.sqrt(
        (loss[0, :, 1] * (1 - loss[0, :, 1])) / data.shape[0] / m),
                     loss[0, :, 1] + 1.96 * np.sqrt(
                         (loss[0, :, 1] * (1 - loss[0, :, 1])) / data.shape[0] / m), color='orange',
                     alpha=0.1)
    plt.fill_between(np.linspace(q + 0.05, 0.95, num), loss[0, :, 2] - 1.96 * np.sqrt(
        (loss[0, :, 2] * (1 - loss[0, :, 2])) / data.shape[0] / m),
                     loss[0, :, 2] + 1.96 * np.sqrt(
                         (loss[0, :, 2] * (1 - loss[0, :, 2])) / data.shape[0] / m), color='green',
                     alpha=0.1)
    plt.fill_between(np.linspace(q + 0.05, 0.95, num), loss[0, :, 3] - 1.96 * np.sqrt(
        (loss[0, :, 3] * (1 - loss[0, :, 3])) / data.shape[0] / m),
                     loss[0, :, 3] + 1.96 * np.sqrt(
                         (loss[0, :, 3] * (1 - loss[0, :, 3])) / data.shape[0] / m), color='red',
                     alpha=0.1)
    plt.legend(loc='upper right')
    plt.title("Hamming loss, (n,q)=(" + str(n) + "," + str(q) + ")")
    plt.xlabel("p")
    plt.ylabel("Hamming loss")
    title = "Hamming_n" + str(n) + "_b" + str(n_batch) + "_num" + str(num) + "_q" + str(q) + ".png"
    plt.savefig(title)

    # 0-1 Loss
    fig1 = plt.figure()
    plt.errorbar(np.linspace(q + 0.05, 0.95, num), loss[1, :, 0], label='MLE_uniform', linestyle="dashed")
    plt.errorbar(np.linspace(q + 0.05, 0.95, num), loss[1, :, 1], label='Majority', linestyle="dashdot")
    plt.errorbar(np.linspace(q + 0.05, 0.95, num), loss[1, :, 2], label='Modal', linestyle="dotted")
    plt.errorbar(np.linspace(q + 0.05, 0.95, num), loss[1, :, 3], label='MLE', linestyle="solid")
    plt.fill_between(np.linspace(q + 0.05, 0.95, num), loss[1, :, 0] - 1.96 * np.sqrt(
        (loss[1, :, 0] * (1 - loss[1, :, 0])) / data.shape[0]),
                     loss[1, :, 0] + 1.96 * np.sqrt(
                         (loss[1, :, 0] * (1 - loss[1, :, 0])) / data.shape[0]), color='b',
                     alpha=0.1)
    plt.fill_between(np.linspace(q + 0.05, 0.95, num), loss[1, :, 1] - 1.96 * np.sqrt(
        (loss[1, :, 1] * (1 - loss[1, :, 1])) / data.shape[0]),
                     loss[1, :, 1] + 1.96 * np.sqrt(
                         (loss[1, :, 1] * (1 - loss[1, :, 1])) / data.shape[0]), color='orange',
                     alpha=0.1)
    plt.fill_between(np.linspace(q + 0.05, 0.95, num), loss[1, :, 2] - 1.96 * np.sqrt(
        (loss[1, :, 2] * (1 - loss[1, :, 2])) / data.shape[0]),
                     loss[1, :, 2] + 1.96 * np.sqrt(
                         (loss[1, :, 2] * (1 - loss[1, :, 2])) / data.shape[0]), color='green',
                     alpha=0.1)
    plt.fill_between(np.linspace(q + 0.05, 0.95, num), loss[1, :, 3] - 1.96 * np.sqrt(
        (loss[1, :, 3] * (1 - loss[1, :, 3])) / data.shape[0]),
                     loss[1, :, 3] + 1.96 * np.sqrt(
                         (loss[1, :, 3] * (1 - loss[1, :, 3])) / data.shape[0]), color='red',
                     alpha=0.1)
    plt.legend(loc='upper right')
    plt.title("0-1 loss, (n,q)=(" + str(n) + "," + str(q) + ")")
    plt.xlabel("p")
    plt.ylabel("0-1 loss")
    title = "ZeroOne_n" + str(n) + "_b" + str(n_batch) + "_num" + str(num) + "_q" + str(q) + ".png"
    plt.savefig(title)
