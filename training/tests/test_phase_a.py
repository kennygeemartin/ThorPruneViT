import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from thorprunevit.data import NIH_LABELS, prepare_nih_dataframe, patient_split, assert_patient_disjoint, pos_weight_from_dataframe, export_splits
from thorprunevit.metrics import multilabel_metrics
from thorprunevit.model import PrunableViT, clone_pruned, count_parameters, save_pruned_model, load_pruned_model
from thorprunevit.importance import collect_importance
from torch.utils.data import DataLoader,TensorDataset

def tiny_df():
    return pd.DataFrame({'Image Index':[f'{i:08d}.png' for i in range(12)],'Finding Labels':['No Finding','Atelectasis','Cardiomegaly']*4,'Patient ID':np.repeat(np.arange(6),2),'View Position':['PA']*12})
def test_label_parsing(tmp_path):
    p=tmp_path/'m.csv'; tiny_df().to_csv(p,index=False); d=prepare_nih_dataframe(p); assert all('label_'+x in d for x in NIH_LABELS); assert d.label_Atelectasis.sum()==4
def test_split_leakage_and_artifacts(tmp_path):
    d=prepare_nih_dataframe((lambda p:(tiny_df().to_csv(p,index=False),p)[1])(tmp_path/'m.csv')); s=patient_split(d,(.5,.2,.3),42); assert_patient_disjoint(s); export_splits(s,tmp_path/'splits',42); assert (tmp_path/'splits/split_manifest.json').exists()
def test_weights_training_only():
    d=pd.DataFrame({'a':[1,0,0,0],'b':[0,0,0,0]}); w=pos_weight_from_dataframe(d,['a','b']); assert w.tolist()==[3,4]
def test_metrics_known():
    y=np.array([[1,0],[0,1]]); a,p=multilabel_metrics(y,np.array([[.9,.1],[.1,.9]]),.5,['a','b']); assert a['exact_match_accuracy']==1 and a['macro_f1']==1
def test_compact_reload(tmp_path):
    m=PrunableViT(image_size=16,patch_size=8,embed_dim=16,depth=1,heads_per_layer=[2],head_dim=8,mlp_dims=[8],num_classes=2,dropout=0); p=clone_pruned(m,[[0]],[[0,1,2,3]]); p.eval(); x=torch.randn(1,3,16,16); y=p(x); save_pruned_model(p,tmp_path); q=load_pruned_model(tmp_path); q.eval(); assert torch.allclose(y,q(x),atol=1e-6); assert count_parameters(q)<count_parameters(m)
def test_importance_shapes():
    m=PrunableViT(image_size=16,patch_size=8,embed_dim=16,depth=1,heads_per_layer=[2],head_dim=8,mlp_dims=[8],num_classes=2,dropout=0); dl=DataLoader(TensorDataset(torch.randn(2,3,16,16),torch.zeros(2,2)),batch_size=2); r=collect_importance(m,dl,torch.nn.BCEWithLogitsLoss(),torch.device('cpu'),1); assert r[0].head_taylor.shape==(2,) and r[0].ffn_taylor.shape==(8,)
def test_latency_cuda_guard():
    if not torch.cuda.is_available():
        import subprocess; r=subprocess.run([sys.executable,str(ROOT/'scripts/measure_latency.py'),'--device','cuda'],capture_output=True,text=True); assert r.returncode==2 and 'CUDA DEVICE REQUIRED' in r.stdout
