from __future__ import annotations
from typing import Dict, Iterable, Sequence
import numpy as np
from scipy import stats


def summarize(values: Sequence[float], confidence=0.95) -> Dict[str,float]:
    a=np.asarray(values,dtype=float); n=len(a); mean=float(a.mean()); sd=float(a.std(ddof=1)) if n>1 else 0.0
    if n>1:
        sem=stats.sem(a); lo,hi=stats.t.interval(confidence,n-1,loc=mean,scale=sem)
    else: lo=hi=mean
    return {'n':n,'mean':mean,'sd':sd,'ci_low':float(lo),'ci_high':float(hi)}


def paired_test(a: Sequence[float], b: Sequence[float]) -> Dict[str,float]:
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    t,p=stats.ttest_rel(a,b)
    d=(a-b).mean()/(a-b).std(ddof=1) if len(a)>1 and (a-b).std(ddof=1)>0 else np.nan
    if len(a)>=2 and np.any(a!=b):
        w,wp=stats.wilcoxon(a,b)
    else: w=wp=np.nan
    return {'n_pairs':len(a),'mean_difference':float((a-b).mean()),'t_statistic':float(t),'t_p':float(p),
            'wilcoxon_statistic':float(w),'wilcoxon_p':float(wp),'paired_cohens_dz':float(d)}
