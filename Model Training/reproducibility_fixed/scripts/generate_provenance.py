import glob
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from common import ROOT
rows=[]
for f in glob.glob(str(ROOT/'results/**/aggregate.csv'),recursive=True):
    p=Path(f); r=pd.read_csv(p).iloc[0]; seed=r.get('seed',''); model=p.parent.name
    for metric,value in r.items():
        if isinstance(value,(int,float)) and pd.notna(value): rows.append({'claim_id':f'{model}:{metric}','experiment':model,'dataset':'NIH','split':'test','model':model,'seed':seed,'metric':metric,'value':value,'source_file':str(p.relative_to(ROOT)),'checkpoint':str((p.parent/'checkpoint.pt').relative_to(ROOT)) if (p.parent/'checkpoint.pt').exists() else str((p.parent/'model.pt').relative_to(ROOT)),'config':'configs/config.yaml','code_entry_point':'scripts','timestamp':datetime.now(timezone.utc).isoformat()})
cols=['claim_id','experiment','dataset','split','model','seed','metric','value','source_file','checkpoint','config','code_entry_point','timestamp']; pd.DataFrame(rows,columns=cols).to_csv(ROOT/'results/RESULT_PROVENANCE.csv',index=False)
