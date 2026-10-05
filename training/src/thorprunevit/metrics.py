from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
                       label_names: Sequence[str] | None = None) -> Tuple[Dict[str,float], List[Dict[str,float]]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    label_acc = (y_pred == y_true).mean(axis=0)
    aggregate = {
        'mean_per_class_accuracy': float(label_acc.mean()),
        'hamming_accuracy': float((y_pred == y_true).mean()),
        'exact_match_accuracy': float(np.all(y_pred == y_true, axis=1).mean()),
        'micro_precision': float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_recall': float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    }
    valid = [i for i in range(y_true.shape[1]) if len(np.unique(y_true[:,i])) > 1]
    if valid:
        aggregate['macro_auroc'] = float(roc_auc_score(y_true[:,valid], y_prob[:,valid], average='macro'))
        aggregate['micro_auroc'] = float(roc_auc_score(y_true[:,valid], y_prob[:,valid], average='micro'))
    else:
        aggregate['macro_auroc'] = np.nan; aggregate['micro_auroc'] = np.nan
    per=[]
    for i in range(y_true.shape[1]):
        tn, fp, fn, tp = confusion_matrix(y_true[:,i], y_pred[:,i], labels=[0,1]).ravel()
        row={
            'label': label_names[i] if label_names else str(i),
            'accuracy': float(label_acc[i]),
            'precision': float(tp/(tp+fp)) if tp+fp else 0.0,
            'sensitivity_recall': float(tp/(tp+fn)) if tp+fn else 0.0,
            'specificity': float(tn/(tn+fp)) if tn+fp else 0.0,
            'f1': float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0,
            'auroc': float(roc_auc_score(y_true[:,i], y_prob[:,i])) if len(np.unique(y_true[:,i]))>1 else np.nan,
        }
        per.append(row)
    return aggregate, per

def masked_multilabel_metrics(y_true, y_prob, threshold=0.5, label_names=None):
    y=np.asarray(y_true,float); p=np.asarray(y_prob,float); valid=np.isfinite(y)&(y!=-1); pred=p>=threshold
    names=label_names or [str(i) for i in range(y.shape[1])]; per=[]
    for i,name in enumerate(names):
        m=valid[:,i]
        if not m.any(): per.append({'label':name,'n_valid':0,'accuracy':np.nan,'precision':np.nan,'sensitivity_recall':np.nan,'specificity':np.nan,'f1':np.nan,'auroc':np.nan}); continue
        yt=y[m,i].astype(int); yp=p[m,i]; yh=pred[m,i].astype(int); tn,fp,fn,tp=confusion_matrix(yt,yh,labels=[0,1]).ravel()
        per.append({'label':name,'n_valid':int(m.sum()),'accuracy':float((yt==yh).mean()),'precision':float(tp/(tp+fp)) if tp+fp else 0.,'sensitivity_recall':float(tp/(tp+fn)) if tp+fn else 0.,'specificity':float(tn/(tn+fp)) if tn+fp else 0.,'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,'auroc':float(roc_auc_score(yt,yp)) if len(np.unique(yt))>1 else np.nan})
    usable=[r for r in per if not np.isnan(r['auroc'])]
    return {'mean_per_class_accuracy':float(np.nanmean([r['accuracy'] for r in per])),'macro_f1':float(np.nanmean([r['f1'] for r in per])),'macro_auroc':float(np.mean([r['auroc'] for r in usable])) if usable else np.nan,'threshold':threshold},per
