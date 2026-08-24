import argparse, json, sys
from pathlib import Path
import pandas as pd
import yaml
from common import ROOT, load_config, split_dir
from thorprunevit.data import NIH_LABELS, CHEXPERT_LABELS, prepare_nih_dataframe, patient_split, export_splits

def placeholder(p): return not p or 'PATH_TO_' in str(p) or '/path/to/' in str(p).lower()
def resolve_image(root,name):
    p=Path(root)/str(name)
    if p.exists(): return p
    return next(Path(root).glob('**/'+str(name)),None) if Path(root).is_dir() else None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs/config.yaml')); ap.add_argument('--seeds',nargs='*',type=int); a=ap.parse_args()
    cfg=load_config(a.config); art=ROOT/'artifacts'; art.mkdir(exist_ok=True); inventory=[]; missing=[]
    nm=Path(cfg['nih']['metadata_csv']); nr=Path(cfg['nih']['image_root'])
    nih={'metadata_csv':str(nm),'image_root':str(nr),'status':'BLOCKED'}
    if placeholder(nm) or not nm.is_file(): missing.append(f'NIH metadata CSV required: {nm}')
    if placeholder(nr) or not nr.is_dir(): missing.append(f'NIH image root required: {nr}')
    if not missing:
        raw=pd.read_csv(nm); required={'Image Index','Finding Labels','Patient ID'}; absent=sorted(required-set(raw.columns))
        if absent: missing.append('NIH metadata columns missing: '+', '.join(absent))
        else:
            view_counts=raw['View Position'].fillna('MISSING').value_counts().to_dict() if 'View Position' in raw else {}
            df=prepare_nih_dataframe(str(nm)); before=len(df)
            if cfg['nih'].get('frontal_only'):
                if 'View Position' not in df: missing.append('NIH frontal filtering requested but View Position column is absent')
                else: df=df[df['View Position'].isin(['PA','AP'])].copy()
            absent_images=[x for x in df['Image Index'] if resolve_image(nr,x) is None]
            nih.update(status='READY' if not absent_images else 'FAILED',records_before_filter=before,images=len(df),patients=int(df['Patient ID'].nunique()),
              view_position_counts=view_counts,frontal_images=len(df),duplicates=int(df.duplicated().sum()),missing_images=len(absent_images),
              no_finding=int(df['Finding Labels'].fillna('').eq('No Finding').sum()),labels=NIH_LABELS,
              prevalence={x:float(df['label_'+x].mean()) for x in NIH_LABELS})
            if absent_images: missing.append(f'NIH referenced images missing: {len(absent_images)} (first: {absent_images[0]})')
            else:
                for seed in (a.seeds or cfg['experiment']['seeds']): export_splits(patient_split(df,tuple(cfg['nih']['split_ratios']),seed),split_dir(seed),seed)
    (art/'nih_dataset_report.json').write_text(json.dumps(nih,indent=2),encoding='utf-8')
    cr=Path(cfg['chexpert']['root']); ct=Path(cfg['chexpert']['train_csv']); cv=Path(cfg['chexpert']['valid_csv']); chex={'root':str(cr),'status':'BLOCKED'}
    c_missing=[]
    for label,p in [('root',cr),('train_csv',ct),('valid_csv',cv)]:
        if placeholder(p) or not p.exists(): c_missing.append(f'CheXpert {label} required: {p}')
    if not c_missing:
        reports={}; unresolved=0
        for name,p in [('train',ct),('valid',cv)]:
            d=pd.read_csv(p); paths=d.get('Path',pd.Series(dtype=str)); bad=sum(resolve_image(cr,x) is None for x in paths)
            unresolved+=bad; reports[name]={'rows':len(d),'patients':int(paths.astype(str).str.extract(r'(patient\d+)',expand=False).nunique()),
              'studies':int(paths.astype(str).str.extract(r'(study\d+)',expand=False).nunique()),'missing_images':bad,
              'available_labels':[x for x in CHEXPERT_LABELS if x in d],
              'uncertain_counts':{x:int((d[x]==-1).sum()) for x in CHEXPERT_LABELS if x in d},
              'missing_label_counts':{x:int(d[x].isna().sum()) for x in CHEXPERT_LABELS if x in d}}
        chex.update(status='READY' if unresolved==0 else 'FAILED',splits=reports,taxonomy=CHEXPERT_LABELS)
        if unresolved: c_missing.append(f'CheXpert referenced images missing: {unresolved}')
    missing.extend(c_missing); (art/'chexpert_dataset_report.json').write_text(json.dumps(chex,indent=2),encoding='utf-8')
    inventory=[{'dataset':'NIH ChestX-ray14','status':nih['status'],'root':str(nr),'metadata':str(nm)},
      {'dataset':'CheXpert','status':chex['status'],'root':str(cr),'metadata':f'{ct};{cv}'}]
    pd.DataFrame(inventory).to_csv(art/'dataset_inventory.csv',index=False)
    mapping=yaml.safe_load((ROOT/'configs/label_mappings.yaml').read_text(encoding='utf-8'))['entries']
    available=set()
    if cv.is_file(): available=set(pd.read_csv(cv,nrows=0).columns)
    for x in mapping: x['column_present']=x.get('chexpert_column') in available if available else None
    pd.DataFrame(mapping).to_csv(art/'chexpert_label_mapping.csv',index=False)
    if missing:
        print('DATASET VALIDATION BLOCKED'); print('\n'.join('- '+x for x in missing)); return 2
    print('DATASET VALIDATION PASSED'); return 0
if __name__=='__main__': sys.exit(main())
