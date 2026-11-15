"""Download/check the immutable public baseline checkpoint, before inference."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'reproduction/lingbot-weights.json').read_text())
target=root/'weights/14b-causal-fast'
target.mkdir(parents=True,exist_ok=True)

def fetch(row):
    p=Path(hf_hub_download('robbyant/lingbot-world-v2-14b-causal-fast',row['path'],
        revision='5c33dd40b213598c418fd25bff30fdbd23fd38a7',local_dir=target))
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):digest.update(data)
    if p.stat().st_size!=row['bytes'] or digest.hexdigest()!=row['sha256']:
        raise ValueError('Pinned checkpoint changed: '+row['path'])
    print('verified',row['path'],flush=True)
with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(fetch,manifest['files']))
link=root/'external/LingBot-World2/checkpoints/lingbot-world-v2-14b-causal-fast'
link.parent.mkdir(parents=True,exist_ok=True)
if not link.exists():link.symlink_to(target)
(root/'outputs').mkdir(exist_ok=True)
(root/'outputs/weights-verified.json').write_text(json.dumps({'status':'verified','files':len(manifest['files']),'revision':'5c33dd40b213598c418fd25bff30fdbd23fd38a7'})+'\n')
