import glob,re
from pathlib import Path
import pandas as pd
from common import ROOT
from thorprunevit.stats import summarize

res=ROOT/'results'; out=res/'reviewer_tables'; out.mkdir(exist_ok=True)
records=[]
for f in glob.glob(str(res/'**'/'aggregate.csv'),recursive=True):
    row=pd.read_csv(f).iloc[0].to_dict(); row['run']=Path(f).parent.name; records.append(row)
if not records:
    raise SystemExit('No real-run aggregate CSV files found. Execute experiments first.')
df=pd.DataFrame(records)

# A compact master table for all experiments.
df.to_csv(out/'all_aggregate_runs.csv',index=False)

# Summarize numeric metrics by run name with seed number normalized.
df['experiment']=df['run'].str.replace(r'_seed_\d+','_seed_*',regex=True)
metrics=['mean_per_class_accuracy','hamming_accuracy','exact_match_accuracy','macro_auroc','micro_auroc','macro_f1','parameters','approx_flops','latency_ms_mean']
rows=[]
for exp,g in df.groupby('experiment'):
    for metric in metrics:
        if metric in g and g[metric].notna().any():
            rows.append({'experiment':exp,'metric':metric,**summarize(g[metric].dropna().values)})
pd.DataFrame(rows).to_csv(out/'summary_mean_sd_ci.csv',index=False)

# Disease-wise merge across seeds for the main 53% joint experiment and baseline.
for pattern,name in [('baseline_seed_*','nih_baseline_disease_wise'),('pruned_seed_*_target_0.53_tw_0.50_mw_0.50_joint','nih_thorprunevit_disease_wise')]:
    files=[]
    for d in res.glob(pattern):
        f=d/'disease_wise.csv'
        if f.exists(): files.append(f)
    if files:
        frames=[]
        for f in files:
            x=pd.read_csv(f); x['seed']=re.search(r'seed_(\d+)',str(f)).group(1); frames.append(x)
        allx=pd.concat(frames,ignore_index=True); allx.to_csv(out/f'{name}_by_seed.csv',index=False)
        summ=[]
        for lab,g in allx.groupby('label'):
            row={'label':lab}
            for m in ['accuracy','precision','sensitivity_recall','specificity','f1','auroc']:
                s=summarize(g[m].dropna().values); row[m+'_mean']=s['mean']; row[m+'_sd']=s['sd']; row[m+'_ci_low']=s['ci_low']; row[m+'_ci_high']=s['ci_high']
            summ.append(row)
        pd.DataFrame(summ).to_csv(out/f'{name}_summary.csv',index=False)
print('Reviewer tables written to',out)
