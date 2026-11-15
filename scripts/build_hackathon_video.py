"""Render the 120-second WorldSync presentation from saved, hash-checked evidence."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/demo'
WORK=ROOT/'outputs/demo-prep'
RUN=ROOT/'outputs/codex-ccv/artifacts/codex-ccv'
EVIDENCE=ROOT/'reproduction/baseline-gap/ccv-snapshot'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
BG='#0b1422'; WHITE='#f3f6fa'; MUTED='#a5b4c8'; GREEN='#61e2b3'; GRAY='#8d9bb0'; PURPLE='#c2a5ff'
W,H,FPS=1280,720,30


def font(n,bold=False):return ImageFont.truetype(BOLD if bold else FONT,n)
def text(draw,xy,s,size=28,fill=WHITE,bold=False):draw.text(xy,s,font=font(size,bold),fill=fill)
def wrap(draw,xy,s,width=85,size=23,fill=MUTED):
    for i,line in enumerate(textwrap.wrap(s,width)):text(draw,(xy[0],xy[1]+i*(size+9)),line,size,fill)
def canvas(kicker):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    d.rectangle((40,38,82,44),fill=GREEN);text(d,(98,27),kicker.upper(),17,MUTED,True)
    text(d,(40,677),'WORLDSYNC  /  LINGBOT-WORLD2',15,MUTED)
    return im,d

def save(im,name):
    p=WORK/(name+'.png');im.save(p);return p

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def episode(arm,task):
    p=RUN/arm/f'if-{task}';return p/'episode' if arm=='C' else p

def score(arm,task):
    video=episode(arm,task)/'rollout.mp4';values=[]
    for rep in ['rep1','rep2','rep3']:
        p=EVIDENCE/'judge'/rep/'episodes'/f'{arm}-if-{task}.json';r=json.loads(p.read_text())
        assert not r.get('error') and not r.get('dry_run') and r['video_sha256']==digest(video)
        value=r['scores']['official'];assert type(value) in (int,float) and 1<=value<=5
        values.append(value)
    run=json.loads((episode(arm,task)/'run.json').read_text())
    assert run['status']=='completed' and run['seed']==42
    return {'scores':values,'mean':sum(values)/3,'video_seconds':run['generated_seconds'],
            'video_sha256':digest(video),'hardware':run['hardware'],'video':str(video.relative_to(ROOT))}

def still_segment(image,seconds,name):
    target=WORK/(name+'.mp4')
    subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-framerate',str(FPS),'-i',str(image),
        '-frames:v',str(seconds*FPS),'-an','-c:v','libx264','-preset','veryfast','-crf','22',
        '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    return target

def comparison_segment(name,title,subtitle,left,right,labels,seconds,speed,footnotes,source_seconds):
    im,d=canvas('SIDE-BY-SIDE / '+name)
    text(d,(40,75),title,43,WHITE,True);text(d,(40,132),subtitle,20,MUTED)
    for x,label,col in [(40,labels[0],GRAY),(648,labels[1],GREEN)]:
        d.rounded_rectangle((x,176,x+592,220),10,fill='#172338');text(d,(x+16,184),label,23,col,True)
    for i,line in enumerate(footnotes):text(d,(40,575+i*30),line,20,MUTED)
    back=save(im,name+'-background');target=WORK/(name+'.mp4')
    graph=['[0:v]fps=30,format=yuv420p[bg]']
    for i in range(2):
        graph.append(f'[{i+1}:v]setpts=(PTS-STARTPTS)/{speed},fps=30,scale=592:330:force_original_aspect_ratio=decrease,pad=592:330:(ow-iw)/2:(oh-ih)/2:color=0x0b1422,setsar=1,tpad=stop_mode=clone:stop_duration={seconds}[v{i}]')
    graph+=['[bg][v0]overlay=40:230[a]','[a][v1]overlay=648:230[b]']
    current='b';extra_inputs=[]
    for i,x in enumerate([40,648]):
        end=source_seconds[i]/speed
        if end < seconds-0.2:
            label=f'PLAYER ENDED AT {source_seconds[i]:.1f}s - FRAME HELD' if name!='memory' else 'FINAL FRAME - HELD FOR COMPARISON'
            nxt='end'+str(i)
            strip=Image.new('RGB',(592,48),BG);strip_draw=ImageDraw.Draw(strip)
            text(strip_draw,(12,13),label,17,WHITE,True)
            strip_path=save(strip,name+'-end-'+str(i));input_index=3+len(extra_inputs)
            extra_inputs.append(strip_path)
            graph.append(f"[{current}][{input_index}:v]overlay={x}:512:enable='gte(t,{end})'[{nxt}]")
            current=nxt
    extra_args=[item for p in extra_inputs for item in ['-loop','1','-framerate','30','-i',str(p)]]
    subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-framerate','30','-i',str(back),'-i',str(left),'-i',str(right),*extra_args,
        '-filter_complex_threads','2','-filter_complex',';'.join(graph),'-map',f'[{current}]','-frames:v',str(seconds*FPS),
        '-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    return target

def main():
    OUT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True)
    tasks=['IF012','IF006','IF022'];rows={t:{a:score(a,t) for a in ['L','C']} for t in tasks}
    means={a:sum(rows[t][a]['mean'] for t in tasks)/3 for a in ['L','C']};gain=means['C']-means['L']
    evidence={'scope':'Exploratory PlayWorld IF subset, not full benchmark; same open weights, no training',
              'selection':'Tasks preselected by hash before scoring; IF highlighted after seeing results',
              'tasks':rows,'means':means,'gain':gain,'replicates_per_video':3,'seed':42,
              'configuration_differences':['reference-guided controller','task-only prompt instead of caption+task','60-chunk duration instead of early END'],
              'preview_policy':'Boardwalk full videos at2x; pond first16seconds at1x; shorter panels explicitly marked when held; scores from original full videos',
              'memory_example':'Separate scripted GC007 long-return test; not the method used for the IF scores'}
    memory_root=ROOT/'reproduction/memory-validation/GC007-seed42-turn_long_wait_return'
    memory_report=json.loads((memory_root/'comparison.json').read_text())
    for arm in ['stock','retrieval']:
        assert digest(memory_root/(arm+'.mp4'))==memory_report[arm]['video_sha256']
    evidence['memory_evidence']={arm:memory_report[arm] for arm in ['stock','retrieval']}
    (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    im,d=canvas('OPEN-WEIGHT WORLD MODELS / HACKATHON DEMO')
    text(d,(40,118),'WorldSync',104,WHITE,True)
    text(d,(44,264),'Make the world follow through.',43,GREEN,True)
    wrap(d,(44,350),'Better task execution with reference-guided control. Longer spatial memory with archived-view retrieval.',50,29)
    text(d,(44,552),'LingBot-World2 14B  |  Frozen weights  |  No retraining',23,MUTED)
    title=save(im,'01-title')
    im,d=canvas('THE METHOD / REFERENCE-GUIDED CONTROL')
    text(d,(40,82),'Keep intent and camera state aligned.',43,WHITE,True)
    cards=[('01','TASK PROMPT','Give the generator the objective.'),('02','PERSISTENT STATE','Track the camera across chunks.'),('03','VISUAL FEEDBACK','Adjust movement from new frames.')]
    for i,(number,heading,body) in enumerate(cards):
        x=40+i*408;d.rounded_rectangle((x,225,x+384,452),18,fill='#172338')
        text(d,(x+24,245),number,45,GREEN,True);text(d,(x+24,319),heading,24,WHITE,True)
        wrap(d,(x+24,367),body,26,21)
    wrap(d,(40,514),'Quarter-second action scheduling and visual hold checks. The world-model weights stay unchanged.',83,25)
    text(d,(40,610),'Evaluated as a complete controller + prompt + duration configuration.',20,MUTED)
    method=save(im,'02-method')
    im,d=canvas('THE RESULT / PLAYWORLD INSTRUCTION-FOLLOWING SUBSET')
    text(d,(40,82),f'{means["L"]:.2f}  →  {means["C"]:.2f}',80,WHITE,True)
    text(d,(660,104),f'+{gain:.2f} points',48,GREEN,True)
    text(d,(44,195),'All 3 preselected tasks improved in this run.',29,WHITE)
    text(d,(44,253),'Mean of 3 Gemini judgments per original video.',21,MUTED)
    for j,t in enumerate(tasks):
        y=338+j*84;l,c=rows[t]['L']['mean'],rows[t]['C']['mean']
        text(d,(44,y-12),t,25,WHITE,True);text(d,(200,y-12),f'{l:.2f} → {c:.2f}',26,GREEN,True)
        d.line((620,y+6,1160,y+6),fill='#26364b',width=3)
        x1=620+(l-1)/4*540;x2=620+(c-1)/4*540
        d.line((x1,y+6,x2,y+6),fill=GREEN,width=6)
        for x,col in [(x1,GRAY),(x2,GREEN)]:d.ellipse((x-8,y-2,x+8,y+14),fill=col)
    for value in range(1,6):text(d,(612+(value-1)*135,584),str(value),18,MUTED)
    text(d,(44,625),'3 tasks • seed 42 • same checkpoint and GPU within each pair • exploratory subset',19,MUTED)
    scores=save(im,'03-scores')
    im,d=canvas('WHAT WE HAVE / WHAT COMES NEXT')
    text(d,(40,84),'A promising starting point.',51,WHITE,True)
    text(d,(40,184),f'+{gain:.2f} on the 3-task IF subset',38,GREEN,True)
    wrap(d,(40,276),'Better task scores with a frozen open model. A separate memory test recovers familiar landmarks after a long absence.',40,26)
    text(d,(40,452),'Next: combine control and memory,',27,WHITE,True)
    text(d,(40,491),'then improve evolving process state.',27,WHITE,True)
    cap=cv2.VideoCapture(str(episode('C','IF006')/'rollout.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,cap.get(cv2.CAP_PROP_FRAME_COUNT)-1);ok,frame=cap.read();cap.release()
    if ok:
        thumb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));thumb.thumbnail((490,310));im.paste(thumb,(750,260))
    text(d,(750,570),'Still breaks: late-frame texture drift.',20,MUTED)
    text(d,(40,623),'Not an overall benchmark win. Full episodes, scores and code are in the repo.',20,MUTED)
    close=save(im,'07-close')
    memory=ROOT/'reproduction/memory-validation/GC007-seed42-turn_long_wait_return'
    jobs=[lambda:still_segment(title,10,'01'),lambda:still_segment(method,15,'02'),lambda:still_segment(scores,17,'03'),
      lambda:comparison_segment('04-boardwalk','Boardwalk: +1.83 PlayWorld points','IF022 / Walk forward along the boardwalk and continue.',
        episode('L','IF022')/'rollout.mp4',episode('C','IF022')/'rollout.mp4',('LEGACY SETUP  1.67','WORLDSYNC SETUP  3.50'),30,2,
        ['Full episodes at 2×. Scores use the original videos; ended panels hold their final frame.',
         'The scene persists longer; continued walking is still only partially completed.'],[rows['IF022'][a]['video_seconds'] for a in ['L','C']]),
      lambda:comparison_segment('05-pond','Pond: steering the character toward the goal','IF006 / The character walks left into the pond.',
        episode('L','IF006')/'rollout.mp4',episode('C','IF006')/'rollout.mp4',('LEGACY SETUP  2.00','WORLDSYNC SETUP  2.56'),16,1,
        ['Excerpt: first 16 seconds at 1×. Scores use the original full episodes.',
         'Later texture artifacts remain; a final-frame example is shown at the end.'],[rows['IF006'][a]['video_seconds'] for a in ['L','C']]),
      lambda:comparison_segment('memory','Separately: recover a place after looking away','GC007 / 31.8-second scripted revisit, including a 24-second hold.',
        memory/'stock.mp4',memory/'retrieval.mp4',('STOCK / DINO 0.574','ARCHIVED-VIEW MEMORY / DINO 0.746'),16,2.5,
        ['Full videos at 2.5×, then final frames held. Retrieval reuses earlier KV features.',
         'Separate memory diagnostic; DINO similarity is not a PlayWorld score.'],[31.8125,31.8125]),
      lambda:still_segment(close,16,'07')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:segments=list(pool.map(lambda job:job(),jobs))
    listing=WORK/'concat.txt';listing.write_text(''.join("file '"+str(p)+"'\n" for p in segments))
    target=OUT/'worldsync-2min.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(target)],check=True)
    # Separate full boardwalk comparison is convenient during Q&A.
    import shutil
    shutil.copy2(segments[3],OUT/'IF022-side-by-side.mp4')
    shutil.copy2(scores,OUT/'subset-scores.png')
    report=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(target)],text=True))
    stream=report['streams'][0];assert int(stream['nb_read_frames'])==3600 and abs(float(stream['duration'])-120)<0.05
    (OUT/'render-verification.json').write_text(json.dumps({'ffprobe':report,'sha256':digest(target),'bytes':target.stat().st_size,'audio':'captioned silent presentation; matching voiceover supplied separately'},indent=2)+'\n')
    print(json.dumps({'video':str(target),'duration':stream['duration'],'frames':stream['nb_read_frames'],'bytes':target.stat().st_size}),flush=True)

if __name__=='__main__':main()
