import glob, os, re
from pathlib import Path
import pandas as pd
from common import ROOT
from thorprunevit.stats import summarize
rows=[]
for f in glob.glob(str(ROOT/'results'/'**'/'aggregate.csv'),recursive=True):
    d=pd.read_csv(f).iloc[0].to_dict(); d['run']=Path(f).parent.name; rows.append(d)
if not rows:
    raise SystemExit('No aggregate.csv files found.')
df=pd.DataFrame(rows); df.to_csv(ROOT/'results'/'all_runs.csv',index=False)
summary=[]
for key,g in df.groupby(df['run'].str.replace(r'_seed_\d+','_seed_*',regex=True)):
    for metric in ['mean_per_class_accuracy','macro_auroc','macro_f1','parameters','approx_flops','latency_ms_mean']:
        if metric in g and g[metric].notna().any(): summary.append({'experiment':key,'metric':metric,**summarize(g[metric].dropna().values)})
pd.DataFrame(summary).to_csv(ROOT/'results'/'summary_mean_sd_ci.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
