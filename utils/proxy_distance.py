import numpy as np
import torch
from sklearn import svm

def compute_proxy_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = source_X.shape[0]
    nb_target = target_X.shape[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    source_idxs = torch.randperm(nb_source)
    target_idxs = torch.randperm(nb_target)
    train_X = torch.cat(
        (source_X[source_idxs[:half_source], :],
         target_X[target_idxs[:half_target], :]), dim=0)
    train_Y = torch.cat(
        (torch.zeros(half_source, dtype=torch.int),
         torch.ones(half_target, dtype=torch.int)), dim=0).numpy()

    test_X = torch.cat(
        (source_X[source_idxs[half_source:], :],
         target_X[target_idxs[half_target:], :]), dim=0)
    test_Y = torch.cat(
        (torch.zeros(nb_source - half_source, dtype=torch.int),
         torch.ones(nb_target - half_target, dtype=torch.int)), dim=0).numpy()

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f'
                  % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)