import numpy as np
import scipy.linalg as sla
import argparse
import scoring
from sklearn.metrics import balanced_accuracy_score



def load_file(score, trial):
    target_score = []
    nontarget_score = []
    score_all = []
    with open(score, 'r') as f:
        lines = f.readlines()
    if lines[0].split()[1] != 'English' and lines[0].split()[1] != '0':
        for line_ in lines:
            score_all.append(float(line_.split()[1].strip()))
            score_all.append(float(line_.split()[-1].strip()))
    else:
        score_all = [float(x.split()[-1].strip()) for x in lines]

    with open(trial, 'r') as f:
        lines = f.readlines()
    label_all = [x.split()[-1].strip() for x in lines]
    for ind_ in range(len(score_all)):
        if label_all[ind_] == 'target':
            target_score.append(score_all[ind_])
        else:
            nontarget_score.append(score_all[ind_])
    score_all = np.array(score_all)
    return np.array(target_score), np.array(nontarget_score), np.reshape(score_all, (2,-1))


def pavx(y):
    """PAV: Pool Adjacent Violators algorithm. Non-paramtetric optimization subject to monotonicity.
     ghat = pav(y)
     fits a vector ghat with nondecreasing components to the
     data vector y such that sum((y - ghat).^2) is minimal.
     (Pool-adjacent-violators algorithm).
    Author: This code is and adaptation from Bosaris Toolkit and
            it is a simplified version of the 'IsoMeans.m' code made available
            by Lutz Duembgen at:
              http://www.imsv.unibe.ch/~duembgen/software
    Args:
     y: uncalibrated scores
    Returns:
      Calibrated scores
      Width of pav bins, from left to right
         (the number of bins is data dependent)
      Height: corresponding heights of bins (in increasing order)
    """
    assert isinstance(y, np.ndarray)

    n = len(y)
    assert n > 0
    index = np.zeros(y.shape, dtype=int)
    l = np.zeros(y.shape, dtype=int)
    # An interval of indices is represented by its left endpoint
    # ("index") and its length "len"
    ghat = np.zeros_like(y)

    ci = 0
    index[ci] = 0
    l[ci] = 1
    ghat[ci] = y[0]
    # ci is the number of the interval considered currently.
    # ghat[ci] is the mean of y-values within this interval.
    for j in range(1, n):
        # a new index intervall, {j}, is created:
        ci = ci + 1
        index[ci] = j
        l[ci] = 1
        ghat[ci] = y[j]
        # while ci >= 1 and ghat[np.maximum(ci-1,0)] >= ghat[ci]:
        while ci >= 1 and ghat[ci - 1] >= ghat[ci]:
            # "pool adjacent violators":
            nw = l[ci - 1] + l[ci]
            ghat[ci - 1] = ghat[ci - 1] + (l[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            l[ci - 1] = nw
            ci = ci - 1

    height = np.copy(ghat[: ci + 1])
    width = l[: ci + 1]

    # Now define ghat for all indices:
    while n >= 1:
        for j in range(index[ci], n):
            ghat[j] = ghat[ci]

        n = index[ci]
        ci = ci - 1

    return ghat, width, height


def compute_rocch(tar_scores, non_scores):
    """Computes ROCCH: ROC Convex Hull.
    Args:
      tar_scores: scores for target trials
      nontar_scores: scores for non-target trials
    Returns:
       pmiss and pfa contain the coordinates of the vertices of the
       ROC Convex Hull.
    """
    assert isinstance(tar_scores, np.ndarray)
    assert isinstance(non_scores, np.ndarray)

    Nt = len(tar_scores)
    Nn = len(non_scores)
    N = Nt + Nn
    scores = np.hstack((tar_scores.ravel(), non_scores.ravel()))
    # ideal, but non-monotonic posterior
    Pideal = np.hstack((np.ones((Nt,)), np.zeros((Nn,))))

    # It is important here that scores that are the same (i.e. already in order) should NOT be swapped.
    # MATLAB's sort algorithm has this property.
    perturb = np.argsort(scores, kind="mergesort")

    Pideal = Pideal[perturb]
    Popt, width, _ = pavx(Pideal)

    nbins = len(width)
    p_miss = np.zeros((nbins + 1,))
    p_fa = np.zeros((nbins + 1,))

    # threshold leftmost: accept eveything, miss nothing
    # 0 scores to left of threshold
    left = 0
    fa = Nn
    miss = 0

    for i in range(nbins):
        p_miss[i] = miss / Nt
        p_fa[i] = fa / Nn
        left = left + width[i]
        miss = np.sum(Pideal[:left])
        fa = N - left - np.sum(Pideal[left:])

    p_miss[nbins] = miss / Nt
    p_fa[nbins] = fa / Nn

    return p_miss, p_fa


def rocch2eer(p_miss, p_fa):
    """Calculates the equal error rate (eer) from pmiss and pfa
    vectors.
    Note: pmiss and pfa contain the coordinates of the vertices of the
    ROC Convex Hull.
    Use compute_rocch to convert target and non-target scores to pmiss and
    pfa values.
    """
    eer = 0

    # p_miss and p_fa should be sorted
    x = np.sort(p_miss, kind="mergesort")
    assert np.all(x == p_miss)
    x = np.sort(p_fa, kind="mergesort")[::-1]
    assert np.all(x == p_fa)

    _1_1 = np.array([1, -1])
    _11 = np.array([[1], [1]])
    for i in range(len(p_fa) - 1):
        xx = p_fa[i : i + 2]
        yy = p_miss[i : i + 2]

        XY = np.vstack((xx, yy)).T
        dd = np.dot(_1_1, XY)
        if np.min(np.abs(dd)) == 0:
            eerseg = 0
        else:
            # find line coefficieents seg s.t. seg'[xx(i)yy(i)] = 1,
            # when xx(i),yy(i) is on the line.
            seg = sla.solve(XY, _11)
            # candidate for EER, eer is highest candidate
            eerseg = 1 / (np.sum(seg))

        eer = np.maximum(eer, eerseg)

    return eer

def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--valid', type=str, help='score file')
    parser.add_argument('--score', type=str, help='score file')
    parser.add_argument('--trial', type=str, help='trial file')
    args = parser.parse_args()

    score_txt = args.score
    trial_txt = args.trial
    scoring.get_trials(args.valid, 2, trial_txt)
    target_score, non_target_score, score_all = load_file(score_txt, trial_txt)
    p_miss, p_fa = compute_rocch(target_score, non_target_score)
    eer = rocch2eer(p_miss, p_fa)

    predictions = np.argmax(score_all,axis=0)
    with open(args.valid) as f:
        lines = f.readlines()
    labels = np.array([int(line_.split()[-1].strip()) for line_ in lines])
    bac = balanced_accuracy_score(labels, predictions)

    print(f"EER : {eer}")
    print(f"BAC : {bac}")
    eer_output_file = 'eer_output.txt'
    with open(eer_output_file, 'a+') as f:
        f.write(f"EER : {eer}\n")
        f.write(f"BAC : {bac}\n")


if __name__ == "__main__":
    main()
