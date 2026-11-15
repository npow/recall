"""Run the declared OE020 text-only ablation pair on one local GPU. Refuses existing output."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]

    out=root/'outputs/local-caption-pair-OE020'
    out.mkdir(exist_ok=False)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    source=out/'source';source.mkdir()
    archive=subprocess.check_output(['git','archive',revision,'worldsync'],cwd=root)
    subprocess.run(['tar','-x','-C',str(source)],input=archive,check=True)
    for name in ['external','reproduction','.env.local']:
     if (root/name).exists():(source/name).symlink_to(root/name,target_is_directory=(root/name).is_dir())
    manifest={'source_commit':revision,'task':'OE020','split':'insight','seed':42,'chunks':60,'arms':['task-only','caption-plus-task'],'status':'running','gpu':'local NVIDIA Thor','purpose':'one paired prompt ablation; not benchmark reproduction','launched_at':time.time(),'jobs':[]}
    def save():
     temp=out/'manifest.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(out/'manifest.json')
    save()
    for arm in manifest['arms']:
     cmd=[str(root/'.venv-lingbot/bin/python'),'-m', 'worldsync.lingbot_released_baseline' if arm=='task-only' else 'worldsync.lingbot_caption_ablation','--task-id','OE020','--seed','42','--chunks','60','--generate','--output',str(out/arm)]
     if arm=='task-only':cmd+=['--split','insight','--validate-controller']
     job={'arm':arm,'command':cmd,'started_at':time.time()};manifest['jobs'].append(job);save()
     with (out/(arm+'.log')).open('w') as log:
      proc=subprocess.Popen(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(source)})
      job['pid']=proc.pid;save();job['returncode']=proc.wait()
     job['finished_at']=time.time();save()
     if job['returncode']:
      manifest['status']='failed';save();sys.exit(job['returncode'])
    manifest['status']='completed';manifest['finished_at']=time.time();save()

if __name__ == "__main__":
    main()
