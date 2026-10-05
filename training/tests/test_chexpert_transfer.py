import subprocess, sys
from pathlib import Path
import numpy as np, torch, yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from thorprunevit.data import map_uncertainty, MaskedBCEWithLogitsLoss, CHEXPERT_LABELS
from thorprunevit.metrics import masked_multilabel_metrics
from evaluate_chexpert_transfer import load_mapping, validate_mapping, evaluate_transfer_arrays

def test_uncertainty_mappings():
    x=torch.tensor([[-1.,0.,1.]])
    assert map_uncertainty(x,'u-zeros')[0].tolist()==[[0,0,1]]
    assert map_uncertainty(x,'u-ones')[0].tolist()==[[1,0,1]]
def test_ignore_masked_bce():
    logits=torch.tensor([[0.,0.]],requires_grad=True); target=torch.tensor([[-1.,1.]])
    loss=MaskedBCEWithLogitsLoss('ignore')(logits,target); assert torch.allclose(loss,torch.tensor(0.6931472),atol=1e-6); loss.backward(); assert logits.grad[0,0]==0 and logits.grad[0,1]!=0
    assert torch.isfinite(MaskedBCEWithLogitsLoss('ignore')(torch.zeros(1,2),torch.full((1,2),-1.)))
def test_masked_metrics():
    a,p=masked_multilabel_metrics(np.array([[1,-1],[0,-1]]),np.array([[.9,.2],[.1,.8]]),label_names=['known','masked']); assert a['mean_per_class_accuracy']==1; assert p[1]['n_valid']==0
def test_mapping_indices_and_exclusions():
    entries=load_mapping(); assert len(entries)==7; assert all(x['nih_label']!='Mass' for x in entries); validate_mapping(entries,14,CHEXPERT_LABELS)
def test_mapping_rejects_output_dimension():
    try: validate_mapping(load_mapping(),2,CHEXPERT_LABELS); assert False
    except ValueError: pass
def test_transfer_saves_raw(tmp_path):
    e=load_mapping(); logits=np.zeros((3,14)); targets=np.zeros((3,14)); evaluate_transfer_arrays(logits,targets,e,tmp_path,42); assert (tmp_path/'raw_predictions.npz').exists() and (tmp_path/'mapping_used.csv').exists()
def test_transfer_refuses_missing_checkpoint():
    r=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_chexpert_transfer.py'),'--seed','42','--checkpoint',str(ROOT/'missing_checkpoint')],capture_output=True,text=True); assert r.returncode==2 and 'checkpoint directory missing' in r.stdout
def test_transfer_refuses_placeholder_paths(tmp_path):
    cp=tmp_path/'checkpoint'; cp.mkdir(); r=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_chexpert_transfer.py'),'--seed','42','--checkpoint',str(cp)],capture_output=True,text=True); assert r.returncode==2 and 'configure CheXpert' in r.stdout
