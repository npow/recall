"""Pin and verify the official small LingBot checkpoint; reuse no other weights."""
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE','1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','600')
from huggingface_hub import HfApi, snapshot_download
ROOT=Path(__file__).resolve().parents[1]
MODEL='robbyant/lingbot-world-v2-1.3b-causal-fast'
REVISION='7e36a5f919f86cb4255cc9bfc30adb44963fbde1'
output=ROOT/'reproduction/lingbot-small'
output.mkdir(exist_ok=True)
info=HfApi().model_info(MODEL,revision=REVISION,files_metadata=True)
files=[x for x in info.siblings if x.rfilename.endswith('.safetensors') or x.rfilename in {'model.safetensors.index.json','README.md'}]
target=ROOT/'external/LingBot-World2/checkpoints/lingbot-world-v2-1.3b-causal-fast'
snapshot_download(MODEL,revision=REVISION,local_dir=target,allow_patterns=[x.rfilename for x in files],max_workers=6)
def verify(entry):
 path=target/entry.rfilename
 h=hashlib.sha256()
 with path.open('rb') as stream:
  while data:=stream.read(8*1024*1024):h.update(data)
 row={'path':entry.rfilename,'bytes':path.stat().st_size,'sha256':h.hexdigest()}
 if entry.lfs and (row['sha256']!=entry.lfs.sha256 or row['bytes']!=entry.size):raise ValueError('Checkpoint integrity failure: '+entry.rfilename)
 if path.suffix=='.safetensors':
  with path.open('rb') as stream:
   size=int.from_bytes(stream.read(8),'little'); header=json.loads(stream.read(size))
  row['dtypes']=sorted({v['dtype'] for k,v in header.items() if k!='__metadata__'})
  import math
  row['tensor_elements']=sum(math.prod(v['shape']) for k,v in header.items() if k!='__metadata__')
 return row
rows=list(ThreadPoolExecutor(max_workers=6).map(verify,files))
report={'model_id':MODEL,'checkpoint_revision':REVISION,'status':'verified','download_workers':6,'path':str(target),'files':rows,'total_bytes':sum(x['bytes'] for x in rows),'total_tensor_elements':sum(x.get('tensor_elements',0) for x in rows),'reuse_assets':'14B checkpoint T5/VAE/tokenizer; not downloaded again'}
(output/'weights.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
