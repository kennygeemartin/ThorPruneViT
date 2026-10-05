# Magnitude-only structured-pruning baseline.
import subprocess, sys
from common import ROOT
subprocess.run([sys.executable,str(ROOT/'scripts/prune_iteratively.py'),'--taylor-weight','0','--magnitude-weight','1',*sys.argv[1:]],check=True)
