import argparse, re
from pathlib import Path
import pandas as pd
from common import ROOT
from thorprunevit.stats import paired_test
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--baseline',default='baseline_seed_*'); ap.add_argument('--comparison',required=True); a=ap.parse_args(); rows=[]
    def collect(pattern):
        z={}
        for d in (ROOT/'results').glob(pattern):
            m=re.search(r'seed_(\d+)',d.name); f=d/'aggregate.csv'
            if m and f.exists(): z[int(m.group(1))]=pd.read_csv(f).iloc[0]
        return z
    b,c=collect(a.baseline),collect(a.comparison); seeds=sorted(set(b)&set(c))
    for metric in ['mean_per_class_accuracy','exact_match_accuracy','macro_auroc','micro_auroc','macro_f1']:
        if seeds and all(metric in b[s] and metric in c[s] for s in seeds): rows.append({'metric':metric,'comparison':a.comparison,**paired_test([c[s][metric] for s in seeds],[b[s][metric] for s in seeds]),'effect_direction':'comparison-baseline'})
    (ROOT/'results').mkdir(exist_ok=True); pd.DataFrame(rows).to_csv(ROOT/'results/statistical_tests.csv',index=False)
if __name__=='__main__': main()
