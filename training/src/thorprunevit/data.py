from __future__ import annotations
import os, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

NIH_LABELS = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia',
    'Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia'
]
CHEXPERT_LABELS = [
    'No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema',
    'Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices'
]


def default_transform(train: bool, image_size: int = 224):
    from torchvision import transforms
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(7)]
    ops += [transforms.ToTensor(), transforms.Normalize([0.485]*3, [0.229]*3)]
    return transforms.Compose(ops)


class NIHChestXray14(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, train: bool = False, image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = default_transform(train, image_size)

    def __len__(self): return len(self.df)

    def _resolve(self, name: str) -> Path:
        p = self.image_root / name
        if p.exists(): return p
        matches = list(self.image_root.glob(f'**/{name}'))
        if not matches: raise FileNotFoundError(name)
        return matches[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self._resolve(row['Image Index'])).convert('RGB')
        y = torch.tensor([float(row[f'label_{x}']) for x in NIH_LABELS], dtype=torch.float32)
        return self.transform(img), y


def prepare_nih_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for lab in NIH_LABELS:
        df[f'label_{lab}'] = df['Finding Labels'].fillna('').str.split('|').apply(lambda xs: float(lab in xs))
    if 'Patient ID' not in df.columns:
        # Stable fallback from filename, only if the metadata lacks Patient ID.
        df['Patient ID'] = df['Image Index'].str.extract(r'(\d+)').astype(int)
    return df


def patient_split(df: pd.DataFrame, ratios=(0.7,0.1,0.2), seed: int = 42):
    rng = np.random.default_rng(seed)
    patients = df['Patient ID'].drop_duplicates().to_numpy()
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(round(ratios[0]*n)); n_val = int(round(ratios[1]*n))
    train_p = set(patients[:n_train]); val_p = set(patients[n_train:n_train+n_val]); test_p=set(patients[n_train+n_val:])
    return (df[df['Patient ID'].isin(train_p)].copy(), df[df['Patient ID'].isin(val_p)].copy(), df[df['Patient ID'].isin(test_p)].copy())


def assert_patient_disjoint(splits):
    ids=[set(x['Patient ID'].tolist()) for x in splits]
    assert ids[0].isdisjoint(ids[1]), 'Patient leakage: train/validation'
    assert ids[0].isdisjoint(ids[2]), 'Patient leakage: train/test'
    assert ids[1].isdisjoint(ids[2]), 'Patient leakage: validation/test'

def export_splits(splits, out_dir: str, seed=None):
    assert_patient_disjoint(splits)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths=[]
    for name, df in zip(['train','val','test'], splits):
        p=Path(out_dir)/f'nih_{name}.csv'; df.to_csv(p,index=False); paths.append(p)
    labels=[f'label_{x}' for x in NIH_LABELS]
    summary=[]; prevalence=[]
    for name,df in zip(['train','val','test'],splits):
        summary.append({'split':name,'patients':df['Patient ID'].nunique(),'images':len(df)})
        for lab in labels: prevalence.append({'split':name,'disease':lab[6:],'positive':int(df[lab].sum()),'prevalence':float(df[lab].mean())})
    pd.DataFrame(summary).to_csv(Path(out_dir)/'split_summary.csv',index=False)
    pd.DataFrame(prevalence).to_csv(Path(out_dir)/'class_prevalence.csv',index=False)
    weights=pos_weight_from_dataframe(splits[0],labels)
    pd.DataFrame({'disease':NIH_LABELS,'positive':splits[0][labels].sum().astype(int).values,
      'negative':(len(splits[0])-splits[0][labels].sum()).astype(int).values,'pos_weight':weights.numpy()}).to_csv(Path(out_dir)/'positive_weights.csv',index=False)
    sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'seed':seed,'date_generated':datetime.now(timezone.utc).isoformat(),'patient_counts':{r['split']:r['patients'] for r in summary},
      'image_counts':{r['split']:r['images'] for r in summary},'sha256':{p.name:sha(p) for p in paths}}
    (Path(out_dir)/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


class CheXpertDataset(Dataset):
    def __init__(self, csv_path: str, root: str, train: bool=False, uncertainty: str='u-ones', image_size: int=224):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.transform = default_transform(train, image_size)
        self.uncertainty = uncertainty

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = Path(str(row['Path']))
        # Typical CSV paths include "CheXpert-v1.0/...". Resolve relative to root.
        if not p.is_absolute():
            if p.parts and p.parts[0].startswith('CheXpert'):
                p = Path(*p.parts[1:])
            p = self.root / p
        img = Image.open(p).convert('RGB')
        vals=[]
        for lab in CHEXPERT_LABELS:
            v = row.get(lab, 0.0)
            if pd.isna(v): v=-1.0 if self.uncertainty == 'ignore' else 0.0
            if float(v) == -1.0:
                if self.uncertainty == 'u-ones': v=1.0
                elif self.uncertainty == 'u-zeros': v=0.0
            vals.append(float(v))
        return self.transform(img), torch.tensor(vals, dtype=torch.float32)


def pos_weight_from_dataframe(df: pd.DataFrame, label_columns: Sequence[str]) -> torch.Tensor:
    y = df[list(label_columns)].to_numpy(dtype=float)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return torch.tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32)


def prepare_chexpert_dataframe(csv_path: str, uncertainty: str='u-ones') -> pd.DataFrame:
    df=pd.read_csv(csv_path)
    for lab in CHEXPERT_LABELS:
        if lab not in df.columns: df[lab]=0.0
        df[lab]=df[lab].fillna(-1.0 if uncertainty=='ignore' else 0.0).astype(float)
        if uncertainty=='u-ones': df.loc[df[lab]==-1.0,lab]=1.0
        elif uncertainty=='u-zeros': df.loc[df[lab]==-1.0,lab]=0.0
    if 'PatientID' not in df.columns:
        # CheXpert path normally contains patientXXXXX.
        df['PatientID']=df['Path'].astype(str).str.extract(r'(patient\d+)',expand=False).fillna(df.index.astype(str))
    return df


def chexpert_patient_split(df: pd.DataFrame, train_ratio: float=0.9, seed: int=42):
    rng=np.random.default_rng(seed); pats=df['PatientID'].drop_duplicates().to_numpy(); rng.shuffle(pats)
    n=int(round(len(pats)*train_ratio)); tr=set(pats[:n]); va=set(pats[n:])
    return df[df['PatientID'].isin(tr)].copy(),df[df['PatientID'].isin(va)].copy()

def map_uncertainty(targets: torch.Tensor, policy: str):
    known=torch.isfinite(targets) & (targets != -1)
    if policy=='u-zeros': return torch.where(known,targets,torch.zeros_like(targets)),torch.ones_like(known,dtype=torch.bool)
    if policy=='u-ones': return torch.where(known,targets,torch.ones_like(targets)),torch.ones_like(known,dtype=torch.bool)
    if policy=='ignore': return torch.where(known,targets,torch.zeros_like(targets)),known
    raise ValueError(f'Unknown uncertainty policy: {policy}')

class MaskedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, policy='ignore', pos_weight=None): super().__init__(); self.policy=policy; self.register_buffer('pos_weight',pos_weight)
    def forward(self, logits, targets):
        mapped,mask=map_uncertainty(targets,self.policy)
        loss=F.binary_cross_entropy_with_logits(logits,mapped,reduction='none',pos_weight=self.pos_weight)
        return (loss*mask).sum()/mask.sum().clamp_min(1)
