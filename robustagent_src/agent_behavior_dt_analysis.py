from bz2 import compress
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree._tree import TREE_LEAF

def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and 
            inner_tree.children_right[index] == TREE_LEAF)

def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:     
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
        is_leaf(inner_tree, inner_tree.children_right[index]) and
        (decisions[index] == decisions[inner_tree.children_left[index]]) and 
        (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        ##print("Pruned {}".format(index))

def prune_duplicate_leaves(mdl):
    """https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier"""
    # Remove leaves if both 
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
    prune_index(mdl.tree_, decisions)


def create_decision_tree(resdata):

    targets = []
    data = []

    operation_end_diffs = []
    job_end_diffs = []
    jobslacks = []
    machineslacks = []
    for d in resdata:
        for ep_data in d:
            action = ep_data['action']
            # if action == 2:
            #     continue
            o = ep_data['obs']
            operation_end_diffs.append(o['operation_end_diff'])
            job_end_diffs.append(o['job_end_diff'])
            jobslacks.append(o['added_slack_job'])
            machineslacks.append(o['added_slack_machine_rel']/o['step_progress'] if o['step_progress'] > 0 else 0)
    mean_op_end_diff = np.mean(operation_end_diffs)
    mean_job_end_diff = np.mean(job_end_diffs)
    mean_jobslack = np.mean(jobslacks)
    mean_machineslack = np.mean(machineslacks)
    

    n_defaults = 0
    n_compresses = 0
    n_strechtes = 0
    for d in resdata:
        for ep_data in d:
            action = ep_data['action']      
            
            defaults = len(list(filter(lambda x: x==0, targets)))
            compresses = len(list(filter(lambda x: x==1, targets)))
            strechtes = len(list(filter(lambda x: x==2, targets)))
            
            if action == 0:
                n_defaults += 1
            if action == 1:
                n_compresses += 1
            if action == 2:
                n_strechtes += 1
                continue

            # if action == 0 and defaults <= compresses and defaults <= strechtes:
            #     targets.append(action)
            # else:
            #     if action == 1 and compresses <= defaults and compresses <= strechtes:
            #         targets.append(action)
            #     else:
            #         if action == 2 and strechtes <= defaults and strechtes <= compresses:
            #             targets.append(action)
            #         else:
            #             continue

            if action == 0 and defaults <= compresses:
                targets.append(action)
            else:
                if action == 1 and compresses <= defaults:
                    targets.append(action)
                else:
                    continue

            o = ep_data['obs']
            data.append(np.array([
                o['critical_path'],
                o['operation_totalslack_above_avg'],
                1 if o['operation_end_diff'] <= mean_op_end_diff else 0,
                1 if o['job_end_diff'] <= mean_job_end_diff else 0,
                1 if o['added_slack_machine_rel'] <= mean_machineslack else 0,
            ]))

    feature_names = np.array(['critical_path', 'totalslack_above_avg', 'operation_end_le_mean', 'job_end_le_mean', 'machine_slack_le_mean'])
    class_names = np.array(['default', 'compress'])
    print(n_defaults)
    print(n_compresses)
    print(n_strechtes)

    X, y = np.array(data), np.array(targets)
    clf = tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=4)
    clf = clf.fit(X, y)
    prune_duplicate_leaves(clf)
    fig = plt.figure(figsize=(20, 20), dpi=100)
    _ = tree.plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
    fig.savefig("agent_behavior_dt.png")
