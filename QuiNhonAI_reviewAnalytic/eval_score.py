import torch
from sklearn.metrics import f1_score
import numpy as np
import datasets

def rss(pred, gt):
    new_pred = []
    new_gt = []
    for i in range(len(pred)):
        if gt[i] == 0 or pred[i] == 0:
            continue
        
        new_pred.append(pred[i])
        new_gt.append(gt[i])
    
    new_pred = np.array(new_pred)
    new_gt = np.array(new_gt)

    if len(new_gt) == 0:
        return 1

    return 1 - np.sum((new_pred - new_gt)**2)/ (16*len(new_gt))

def final_score(gt, pred):
    scores = []
    for j in range(0,6):
        pred_rss, gt_rss = [], []
        pred_f1, gt_f1 = [], []
        
        for i in range(j,len(pred),6):
            pred_rss.append(pred[i])
            gt_rss.append(gt[i])

            pred_f1.append(int(pred[i] > 0))
            gt_f1.append(int(gt[i] > 0))
        
        scores.append(rss(pred_rss, gt_rss)*f1_score(gt_f1, pred_f1, average='binary'))
    assert len(scores) == 6, "ERROR number of prediction scores"
    return np.mean(scores)

class RA_score(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description='_DESCRIPTION',
            citation='_CITATION',
            inputs_description='_KWARGS_DESCRIPTION',
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )

    def _compute(self, predictions, references):
        return {
            "ra_score": float(
                final_score(references, predictions)
            )
        }

if __name__ == "__main__":
    pred = [1, 0, 3, 4, 2, 0, 0, 2, 2, 5, 1, 0]
    gt = [1, 1, 3, 5, 3, 1, 0, 0, 3, 4, 0, 0]
    print(final_score(gt, pred))

    print(f1_score([0, 0, 0, 1], [0, 0, 0, 1], average= 'binary', pos_label=1))