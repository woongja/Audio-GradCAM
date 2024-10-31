import numpy as np

# def compute_det_curve(target_scores, nontarget_scores):

#     n_scores = target_scores.size + nontarget_scores.size
#     all_scores = np.concatenate((target_scores, nontarget_scores))
#     labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

#     # Sort labels based on scores
#     indices = np.argsort(all_scores, kind='mergesort')
#     labels = labels[indices]

#     # Compute false rejection and false acceptance rates
#     tar_trial_sums = np.cumsum(labels)
#     nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

#     frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
#     far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
#     thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

#     return frr, far, thresholds

# def calculate_confusion_matrix(target_scores, nontarget_scores, threshold):
#     """
#     Calculate the confusion matrix for a given threshold.
#     return: tp, tn, fp, fn
#     """
#     tp = np.sum(target_scores > threshold)
#     tn = np.sum(nontarget_scores <= threshold)
#     fn = np.sum(target_scores <= threshold)
#     fp = np.sum(nontarget_scores > threshold)
#     return tp, tn, fp, fn

# def compute_eer(target_scores, nontarget_scores):
#     """ Returns equal error rate (EER) and the corresponding threshold. """
#     frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
#     abs_diffs = np.abs(frr - far)
#     min_index = np.argmin(abs_diffs)
#     eer = np.mean((frr[min_index], far[min_index]))
#     return eer, thresholds[min_index]

# First, we need some functions to compute EER and sasv-eer

def compute_det_curve(target_scores, nontarget_scores):
    """
    frr, far, thr = compute_det_curve(target_scores, nontarget_scores)

    input
    -----
      target_scores:    np.array, target trial scores
      nontarget_scores: np.array, nontarget trial scores

    output
    ------
      frr:   np.array, FRR, (#N, ), where #N is total number of scores + 1
      far:   np.array, FAR, (#N, ), where #N is total number of scores + 1
      thr:   np.array, threshold, (#N, )

    """

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size),
                             np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = (nontarget_scores.size -
                            (np.arange(1, n_scores + 1) - tar_trial_sums))

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums/target_scores.size))
    # false rejection rates
    far = np.concatenate((np.atleast_1d(1),
                          nontarget_trial_sums / nontarget_scores.size))
    # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001),
                                 all_scores[indices]))
    # Thresholds are the sorted scores
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """
    eer, eer_threshold = compute_eer(target_scores, nontarget_scores)

    input
    -----
      target_scores:    np.array, or list of np.array, target trial scores
      nontarget_scores: np.array, or list of np.array, nontarget trial scores

    output
    ------
      eer:            float, EER
      eer_threshold:  float, threshold corresponding to EER

    """
    if type(target_scores) is list and type(nontarget_scores) is list:
        frr, far, thr = compute_det_curve_sets(target_scores, nontarget_scores)
    else:
        frr, far, thr = compute_det_curve(target_scores, nontarget_scores)

    # find the operation point for EER
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)

    # compute EER
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thr[min_index]


def compute_sasv_eer(scores, labels, tar_label = 1, nontar_label = 2, spoof_label = 0):
    """save_eer = compute_sasv_eer(scores, labels)

    input
    -----
      scores:    np.array, (N, ), fused scores for SASV
      labels:    np.array, (N, ), labels

    output
    ------
      save_eer:  float, SASV-EER
    """
    print(type(np.where(labels == tar_label)))
    print(np.where(labels == tar_label))
    tar_scores = scores[np.where(labels == tar_label)]
    other_scores = scores[np.where(labels != tar_label)]
    nontar_scores = scores[np.where(labels == nontar_label)]
    spoof_scores = scores[np.where(labels == spoof_label)]

    sasv_eer, sasv_thr = compute_eer(tar_scores, other_scores)
    return sasv_eer, sasv_thr