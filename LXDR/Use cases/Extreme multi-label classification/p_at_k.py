from tqdm import tqdm
import numpy as np

def precision_at_k_score(y_true, y_pred_proba):
    top1 = []
    top3 = []
    top5 = []
    top1thr = []
    top3thr = []
    top5thr = []
    top1thr2 = []
    top3thr2 = []
    top5thr2 = []
    for y_true_, y_pred_proba_ in tqdm(zip(y_true, y_pred_proba)):
        ranked_y_pred, ranked_y_true = zip(*sorted(zip(y_pred_proba_, y_true_), reverse=True))

        ltop1 = np.multiply(np.array(ranked_y_true[:1]), np.array([1 for i in ranked_y_pred[:1]])).sum()
        top1.append(ltop1 / 1)

        ltop3 = np.multiply(np.array(ranked_y_true[:3]), np.array([1 for i in ranked_y_pred[:3]])).sum()
        top3.append(ltop3 / 3)

        ltop5 = np.multiply(np.array(ranked_y_true[:5]), np.array([1 for i in ranked_y_pred[:5]])).sum()
        top5.append(ltop5 / 5)

        ltop1thr = np.multiply(np.array(ranked_y_true[:1]),
                               np.array([1 if i > 0 else 0 for i in ranked_y_pred[:1]])).sum()
        top1thr.append(ltop1thr / 1)

        ltopk3thr = np.multiply(np.array(ranked_y_true[:3]),
                                np.array([1 if i > 0 else 0 for i in ranked_y_pred[:3]])).sum()
        top3thr.append(ltopk3thr / 3)

        ltop5thr = np.multiply(np.array(ranked_y_true[:5]),
                               np.array([1 if i > 0 else 0 for i in ranked_y_pred[:5]])).sum()
        top5thr.append(ltop5thr / 5)

        ltop1thr2 = np.multiply(np.array(ranked_y_true[:1]),
                                np.array([1 if i > 0.5 else 0 for i in ranked_y_pred[:1]])).sum()
        top1thr2.append(ltop1thr2 / 1)

        ltopk3thr2 = np.multiply(np.array(ranked_y_true[:3]),
                                 np.array([1 if i > 0.5 else 0 for i in ranked_y_pred[:3]])).sum()
        top3thr2.append(ltopk3thr2 / 3)

        ltop5thr2 = np.multiply(np.array(ranked_y_true[:5]),
                                np.array([1 if i > 0.5 else 0 for i in ranked_y_pred[:5]])).sum()
        top5thr2.append(ltop5thr2 / 5)

    print('1:', np.sum(top1) / len(y_true))
    print('3:', np.sum(top3) / len(y_true))
    print('5:', np.sum(top5) / len(y_true))
    print('1 thr:', np.sum(top1thr) / len(y_true))
    print('3 thr:', np.sum(top3thr) / len(y_true))
    print('5 thr:', np.sum(top5thr) / len(y_true))
    print('1 thr:', np.sum(top1thr2) / len(y_true))
    print('3 thr:', np.sum(top3thr2) / len(y_true))
    print('5 thr:', np.sum(top5thr2) / len(y_true))
