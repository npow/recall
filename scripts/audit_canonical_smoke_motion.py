"""Repeat the conservative visual-motion diagnostic on the saved canonical smoke."""
import json
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from worldsync.lingbot_visual_pose import estimate_rotation


def main():
    cv2.setNumThreads(4)
    directory=ROOT/'reproduction/baseline-gap/gpu-smoke-GC010'
    video=directory/'episode/rollout.mp4'
    cap=cv2.VideoCapture(str(video));frames=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames)!=29:raise ValueError('Expected the29-frame smoke artifact')
    height,width=frames[0].shape[:2]
    values=np.load(ROOT/'external/LingBot-World2/examples/03/intrinsics.npy')[0]*np.array([width/832,height/480,width/832,height/480])
    intrinsics=np.array([[values[0],0,values[2]],[0,values[1],values[3]],[0,0,1]])
    rows=[{'from_frame':index,'to_frame':index+4,**estimate_rotation(frames[index],frames[index+4],intrinsics)}
          for index in range(0,28,4)]
    report=json.loads((directory/'episode/run.json').read_text())
    result={'scope':'Post-hoc visual rotation diagnostic, not ground-truth pose or benchmark success',
            'assumptions':'Nominal model intrinsics, rigid pure rotation; generated geometry changes can invalidate estimates',
            'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
            'pairs':rows,'accepted_pairs':sum(row['usable'] for row in rows),'total_pairs':len(rows),
            'whole_yaw_estimate':sum(row['apparent_yaw_degrees'] for row in rows) if all(row['usable'] for row in rows) else None,
            'commanded_final_yaw':report['actions'][-1]['pose_after']['accumulated_yaw_degrees'],'return_tested':False}
    (directory/'visual-motion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'}))


if __name__=='__main__':main()
