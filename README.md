### Readme File:

#Tracking a Composite Truth from Diversely Competent Voters

Epistemic voting interprets votes as noisy signals about a ground truth. When truth consists of a set of objective winners (or true answers to a set of binary issues), possibilities become richer. We offer a study of epistemic multi-winner voting under two hypotheses of growing complexity: under the first one, both voters' probability of casting true and false positives and alternatives' prior probabilities of being among winners  are known a priori, or learnt from a partially known ground truth; under the second one, the discovery of the ground truth and the estimation of the parameters are intertwined in an iterative approach. Orthogonally, we distinguish between situations where the number of objective winners is  constrained (bounded or fixed) or unconstrained. We perform a theoretical analysis as well as experiments from real and synthetic data. Namely, we apply our model to a multi-label keyword predictions in a crowdsourcing scenario.


## Data

The dataset containing the real annotations that we used are publically available at : https://github.com/CrowdTruth/Events-in-videos.

##Code

This repository contains all the code for the implementation of the algorithms and methods described in our paper for the aggregation of approval ballots in multi-winner settings: See the files:
1-main.py: which contains the data pre-processing step, and the code lines for the execution of the experiments.
2-Sim_fun.py: which contains the functions needed for generating synthetic votes in many settings and applying and comparing different aggregation rules for the recovery of the ground truth.
3-Real_exp1.py: which contains the functions that implement the Algorithm 1 and other methods for the aggregation of the real annotations of the dataset.
