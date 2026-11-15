"""Bounded single-H100 baseline lease. Credentials remain in pass/process memory."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://cloud.lambda.ai/api/v1/'


def session():
    key = subprocess.check_output(['pass', 'show', 'api/lambda-cloud'], text=True).splitlines()[0].strip()
    for prefix in ['LAMBDA_API_KEY=', 'api_key:', 'API_KEY=']:
        if key.startswith(prefix):
            key = key[len(prefix):].strip()
    client = requests.Session()
    client.auth = (key, '')
    return client


def call(client, method, path, body=None):
    response = client.request(method, BASE+path, json=body, timeout=45)
    if not response.ok:
        raise RuntimeError(f'Lambda {method} {path}: HTTP{response.status_code}')
    return response.json()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch','watch','status','terminate'])
    parser.add_argument('--lease', type=Path, required=True)
    parser.add_argument('--minutes', type=int, default=60)
    args = parser.parse_args()
    client = session()
    args.lease.parent.mkdir(parents=True,exist_ok=True)
    if args.action == 'launch':
        if args.lease.exists() or not 1 <= args.minutes <= 60:
            raise ValueError('Use a new lease file and at most60minutes')
        existing = call(client,'GET','instances')['data']
        if any(x.get('name','').startswith('worldsync-baseline-') for x in existing):
            raise ValueError('A baseline lease already exists; inspect it before launching another')
        types = call(client,'GET','instance-types')['data']
        candidate = types['gpu_1x_h100_pcie']
        if candidate['instance_type']['price_cents_per_hour'] > 329:
            raise ValueError('Price exceeded checked329cents/hour')
        regions = [r['name'] for r in candidate['regions_with_capacity_available']]
        if 'us-west-3' not in regions:
            raise ValueError('Checked region no longer has capacity; no automatic larger rental')
        name = 'worldsync-baseline-' + str(time.time_ns())
        public = Path('/home/npow/.ssh/id_ed25519.pub').read_text().strip()
        key = call(client,'POST','ssh-keys',{'name':name,'public_key':public})['data']
        started = time.time()
        result = call(client,'POST','instance-operations/launch',{
            'region_name':'us-west-3','instance_type_name':'gpu_1x_h100_pcie',
            'ssh_key_names':[name],'quantity':1,'name':name})
        ids = result['data']['instance_ids']
        if len(ids) != 1:
            raise ValueError('Unexpected launch count')
        lease = {'instance_id':ids[0],'name':name,'ssh_key_name':name,'ssh_key_id':key.get('id'),
                 'instance_type':'gpu_1x_h100_pcie','region':'us-west-3','price_cents_per_hour':329,
                 'created_at_unix':started,'terminate_after_unix':started+60*args.minutes,
                 'maximum_lease_minutes':args.minutes,'status':'launched','purpose':'Corrected stock baseline validation; no algorithm hillclimbing'}
        save(args.lease,lease)
        with args.lease.with_suffix('.watch.log').open('a') as log:
            watcher = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'watch','--lease',str(args.lease.resolve())],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        lease['watchdog_pid']=watcher.pid;save(args.lease,lease)
        print(json.dumps({k:lease[k] for k in ['instance_id','instance_type','price_cents_per_hour','maximum_lease_minutes','watchdog_pid']}))
        return
    lease=json.loads(args.lease.read_text())
    if args.action == 'watch':
        while time.time() < lease['terminate_after_unix']:
            if json.loads(args.lease.read_text()).get('status') == 'terminated':
                return
            time.sleep(min(30,max(1,lease['terminate_after_unix']-time.time())))
        args.action='terminate'
    if args.action == 'terminate':
        while True:
            try:
                rows=call(client,'GET','instances')['data']
                owned=[r for r in rows if r['id']==lease['instance_id']]
                if not owned or owned[0]['status'] in {'terminating','terminated'}:
                    break
                call(client,'POST','instance-operations/terminate',{'instance_ids':[lease['instance_id']]})
                break
            except (RuntimeError, requests.RequestException):
                print('Termination API temporarily unavailable; retrying owned lease',flush=True)
                time.sleep(15)
        lease.update(status='terminated',termination_requested_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(args.lease,lease);print('Termination requested for owned baseline lease')
    elif args.action == 'status':
        row=call(client,'GET','instances/'+lease['instance_id'])['data']
        lease.update(status=row['status'],ip=row.get('ip'));save(args.lease,lease)
        print(json.dumps({'id':row['id'],'status':row['status'],'ip':row.get('ip')}))


if __name__ == '__main__':
    main()
