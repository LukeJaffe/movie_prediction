#!/usr/bin/env python3

import os
import json
from scipy.stats import pearsonr
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np

dataset_dir = './movie-trailers-dataset'
metadata_dir = os.path.join(dataset_dir, 'metadata')
### In metadata dir
# Scalar data
budget_path = os.path.join(metadata_dir, 'Budget.txt')
exp_path = os.path.join(metadata_dir, 'Actor_Experience.txt')
runtime_path = os.path.join(metadata_dir, 'Runtime.txt')
screens_path = os.path.join(metadata_dir, 'Screens.txt')
sequel_path = os.path.join(metadata_dir, 'Sequel.txt')
# Binary vector data
ratings_path = os.path.join(metadata_dir, 'MPAA_Ratings.txt')
genre_path = os.path.join(metadata_dir, 'Genre.txt')
release_path = os.path.join(metadata_dir, 'Release_Period.txt')
### In base dir
gross_path = os.path.join(dataset_dir, 'Opening_Weekend_Gross.txt')
rt_path = os.path.join(dataset_dir, 'rt.json')

def load_scalar(file_path, black_list):
    data_list = []
    with open(file_path, 'r') as fp:
        for i, line in enumerate(fp):
            if i not in black_list:
                data_list.append(float(line.strip()))
    return data_list

def load_binvec(file_path, black_list):
    data_list = []
    with open(file_path, 'r') as fp:
        for i, line in enumerate(fp):
            if i not in black_list:
                vals = [int(x) for x in line.strip().split(',')]
                data_list.append(vals)
    return data_list

budget_list, gross_list, exp_list = [], [], []
rt_class_list, rt_score_list = [], []
blacklist_idx = []
with open(rt_path, 'r') as fp:
    rt_data = json.load(fp)
    for i, d in enumerate(rt_data):
        try:
            c, s = d
        except:
            blacklist_idx.append(i)
        else:
            rt_class_list.append(c)
            rt_score_list.append(s)

budget_list = load_scalar(budget_path, blacklist_idx)
gross_list = load_scalar(gross_path, blacklist_idx)
exp_list = load_scalar(exp_path, blacklist_idx)
runtime_list = load_scalar(runtime_path, blacklist_idx)
screens_list = load_scalar(screens_path, blacklist_idx)
sequel_list = load_scalar(sequel_path, blacklist_idx)

ratings_list = load_binvec(ratings_path, blacklist_idx)
genre_list = load_binvec(genre_path, blacklist_idx)
release_list = load_binvec(release_path, blacklist_idx)

fresh_prior = np.sum(rt_class_list)/len(rt_class_list)
print('Fresh RT prior: {:.3f}'.format(fresh_prior))
print('Rotten RT prior: {:.3f}'.format(1.0-fresh_prior))

c, p = pearsonr(budget_list, gross_list)
print('Correlation between budget and gross: {:.3f}, p={:.3f}'.format(c, p))

c, p = pearsonr(exp_list, budget_list)
print('Correlation between experience and gross: {:.3f}, p={:.3f}'.format(c, p))

c, p = pearsonr(budget_list, rt_score_list)
print('Correlation between budget and RT: {:.3f}, p={:.3f}'.format(c, p))

# Prep features
X = np.array([
    budget_list, 
    gross_list, 
    exp_list, 
    runtime_list, 
    screens_list,
    sequel_list
    ]).T
X = np.concatenate([X, ratings_list], axis=1)
X = np.concatenate([X, genre_list], axis=1)
X = np.concatenate([X, release_list], axis=1)
Y = np.array(rt_class_list)

num_test = 10
num_trials = 100
acc_list = []
for s in range(num_trials):
    # Seed RNG
    np.random.seed(s)

    # Shuffle data
    rand_idx = np.arange(len(X))
    np.random.shuffle(rand_idx)
    X_all = X[rand_idx]
    Y_all = Y[rand_idx]

    # Partition into train and test
    X_train = X_all[:-num_test]
    Y_train = Y_all[:-num_test]
    X_test = X_all[-num_test:]
    Y_test = Y_all[-num_test:]

    # Fit model
    #clf = tree.DecisionTreeClassifier()
    #clf = svm.SVC()
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    acc_list.append((Y_pred==Y_test).sum())

print('Accuracy mean: {:.3f}'.format(np.mean(acc_list)/num_test))
