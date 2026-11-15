"""Build the silent 120-second research presentation from recorded experiments."""
import concurrent.futures
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw
import build_hackathon_video as v

OUT=v.ROOT/'docs/demo'
v.WORK=v.ROOT/'outputs/academic-presentation'
v.WORK.mkdir(parents=True,exist_ok=True)


def slide(section,title,subtitle=None):
    im,d=v.canvas(section)
    v.text(d,(40,78),title,44,v.WHITE,True)
    if subtitle:v.text(d,(42,143),subtitle,22,v.MUTED)
    return im,d


def box(d,rect,heading,body,color=v.GREEN):
    x,y,x2,y2=rect
    d.rounded_rectangle(rect,16,fill='#172338',outline='#2a3a50',width=2)
    v.text(d,(x+18,y+17),heading,24,color,True)
    v.wrap(d,(x+18,y+61),body,max(10,int((x2-x-36)/12)),22,v.WHITE)


def arrow(d,start,end,color=v.GREEN):
    import math
    d.line((start,end),fill=color,width=4)
    a=math.atan2(end[1]-start[1],end[0]-start[0])
    pts=[end,(end[0]-14*math.cos(a-.5),end[1]-14*math.sin(a-.5)),(end[0]-14*math.cos(a+.5),end[1]-14*math.sin(a+.5))]
    d.polygon(pts,fill=color)


def video(arm,split,task):
    p=v.RUN/arm/f'{split}-{task}'
    return p/('episode/rollout.mp4' if arm=='C' else 'rollout.mp4')


def quadrants():
    im,d=slide('03 / CHALLENGES','Four dimensions. Four ways a world breaks.','Recorded LingBot-World2 outputs • corrected configuration • full episodes at 3×')
    cases=[('gc','GC024','GEOMETRY CONSISTENCY','Returning views change appearance.'),
           ('if','IF022','INSTRUCTION FOLLOWING','The walk stops before the task is complete.'),
           ('insight','OE066','IN-SIGHT EVOLUTION','A piping tool morphs; the process fails.'),
           ('outsight','OE062','OUT-OF-SIGHT EVOLUTION','The cup loses identity; the process is lost.')]
    paths=[];graph=['[0:v]fps=30,format=yuv420p[b0]'];sources=[]
    for i,(split,task,head,caption) in enumerate(cases):
        x=40+(i%2)*608;y=191+(i//2)*230
        v.text(d,(x,y),head+' / '+task,19,v.GREEN,True)
        v.text(d,(x,y+201),caption,18,v.WHITE)
        p=video('C',split,task);paths.append(p)
        judge=json.loads((v.EVIDENCE/'judge/rep1/episodes'/f'C-{split}-{task}.json').read_text())
        assert v.digest(p)==judge['video_sha256']
        sources.append({'task':task,'split':split,'video':str(p.relative_to(v.ROOT)),'sha256':v.digest(p),'excerpt':'full59.8125s episode at3x'})
        graph += [f'[{i+1}:v]setpts=(PTS-STARTPTS)/3,fps=30,tpad=stop_mode=clone:stop_duration=1,scale=592:170:force_original_aspect_ratio=decrease,pad=592:170:(ow-iw)/2:(oh-ih)/2:color=0x0b1422,setsar=1[q{i}]',f'[b{i}][q{i}]overlay={x}:{y+28}[b{i+1}]']
    bg=v.save(im,'03-quadrants');target=v.WORK/'03.mp4'
    args=['ffmpeg','-v','error','-y','-loop','1','-framerate','30','-i',str(bg)]
    for p in paths:args+=['-i',str(p)]
    subprocess.run(args+['-filter_complex_threads','2','-filter_complex',';'.join(graph),'-map','[b4]','-frames:v','600','-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-pix_fmt','yuv420p',str(target)],check=True)
    return target,sources


def main():
    rows={t:{a:v.score(a,t) for a in ['L','C']} for t in ['IF012','IF006','IF022']}
    means={a:sum(r[a]['mean'] for r in rows.values())/3 for a in ['L','C']}
    im,d=slide('01 / RESEARCH QUESTION','WorldSync','Long-horizon consistency in interactive world models')
    v.wrap(d,(42,227),'How long can a generated world hold together while an agent acts inside it?',44,42,v.WHITE)
    box(d,(42,443,620,596),'THE PROBLEM','A video can look plausible yet forget a place, ignore an action, or reset a process.')
    box(d,(644,443,1238,596),'OUR APPROACH','Persistent control state and retrieval of earlier visual features. No training from scratch.')
    intro=v.save(im,'01-problem')
    im,d=slide('02 / TASK & BENCHMARK','PlayWorld: act, wait, return, observe.','Open starting model: LingBot-World2 14B causal-fast • frozen weights')
    for x,y,h,body in [(42,205,'Geometry consistency','Does the same place survive a revisit?'),(652,205,'Instruction following','Does the requested action actually happen?'),(42,370,'In-sight evolution','Do visible processes progress coherently?'),(652,370,'Out-of-sight evolution','Does a hidden process continue while we look away?')]:box(d,(x,y,x+586,y+145),h,body)
    v.text(d,(44,554),'Our comparison: 12 selected tasks × 2 setups × 3 Gemini judge attempts.',24,v.WHITE)
    v.text(d,(44,600),'One seed; full original episodes scored. Local configuration, not the hosted leaderboard.',20,v.MUTED)
    setup=v.save(im,'02-setup')
    im,d=slide('04 / METHOD & ARCHITECTURE','Keep control state. Retrieve visual history.','Two separately evaluated interventions on a frozen world model')
    box(d,(40,207,281,338),'TASK + PLAN','Quarter-second action schedule')
    box(d,(335,207,637,338),'STATEFUL CONTROL','Persistent pose + visual hold checks')
    box(d,(695,207,987,338),'WORLD MODEL','Camera rays + text → next frames')
    box(d,(1040,207,1240,338),'OUTPUT','Generated frames')
    for a,b in [((282,270),(331,270)),((638,270),(691,270)),((988,270),(1036,270))]:arrow(d,a,b)
    d.line([(1140,340),(1140,380),(486,380)],fill=v.GREEN,width=3);arrow(d,(486,380),(486,341))
    v.text(d,(434,388),'latest frame → visual feedback',21,v.GREEN)
    box(d,(40,447,377,578),'ARCHIVED VIEWS','Store earlier attention keys and values',v.PURPLE)
    box(d,(430,447,787,578),'RETRIEVE ON RETURN','Insert relevant history into a bounded cache',v.PURPLE)
    arrow(d,(378,512),(426,512),v.PURPLE)
    d.line([(788,512),(841,512),(841,433)],fill=v.PURPLE,width=3);arrow(d,(841,433),(841,340),v.PURPLE)
    v.wrap(d,(903,452),'Memory evaluated separately on a scripted revisit.',24,23,v.PURPLE)
    v.text(d,(42,615),'Control result changes controller + prompt + duration. Memory result is a separate ablation.',20,v.MUTED)
    arch=v.save(im,'04-architecture')
    im,d=slide('05 / QUANTITATIVE RESULTS','Better instruction following on three tasks.','Three preselected IF tasks • three judgments per original video • seed 42')
    v.text(d,(42,207),f'{means["L"]:.2f} → {means["C"]:.2f}',76,v.GREEN,True)
    v.text(d,(760,229),f'+{means["C"]-means["L"]:.2f} points',49,v.GREEN,True)
    for i,(task,r) in enumerate(rows.items()):
        y=329+i*60;v.text(d,(45,y),task,27,v.WHITE,True);v.text(d,(285,y),f'{r["L"]["mean"]:.2f} → {r["C"]["mean"]:.2f}',27,v.WHITE)
        v.text(d,(711,y),f'+{r["C"]["mean"]-r["L"]["mean"]:.2f}',27,v.GREEN)
    v.wrap(d,(42,535),'Exploratory: IF was highlighted after reviewing results. Other dimensions regress; this is not an overall benchmark win.',93,23)
    v.text(d,(42,612),'Same checkpoint and GPU within each pair. No weight updates.',21,v.MUTED)
    results=v.save(im,'05-results')
    im,d=slide('08 / LIMITATIONS & FUTURE WORK','The world still breaks.','Evidence of useful components—not a solved world simulator')
    box(d,(42,211,624,477),'REMAINING FAILURES','Partial action completion. Appearance drift. Foreground objects morph. Hidden processes reset.')
    box(d,(650,211,1238,477),'NEXT EXPERIMENTS','Isolate caption conditioning. Combine control and retrieval. Track evolving process state. Test held-out seeds.')
    v.wrap(d,(44,514),'Takeaway: persistent control improves our IF subset; archived visual memory improves a separate revisit diagnostic.',80,27,v.GREEN)
    v.text(d,(44,614),'Small samples, one seed, noisy judges. Full videos and raw evidence are in the repository.',20,v.MUTED)
    close=v.save(im,'08-future')
    memory=v.ROOT/'reproduction/memory-validation/GC007-seed42-turn_long_wait_return'
    mr=json.loads((memory/'comparison.json').read_text())
    for a in ['stock','retrieval']:assert v.digest(memory/(a+'.mp4'))==mr[a]['video_sha256']
    jobs=[lambda:v.still_segment(intro,10,'01'),lambda:v.still_segment(setup,12,'02'),lambda:quadrants(),lambda:v.still_segment(arch,20,'04'),lambda:v.still_segment(results,12,'05'),
          lambda:v.comparison_segment('06-results-video','IF022: longer execution, higher task score','Continue walking along the boardwalk. Full episodes at 3×.',v.episode('L','IF022')/'rollout.mp4',v.episode('C','IF022')/'rollout.mp4',('LEGACY SETUP / 1.67','CORRECTED SETUP / 3.50'),20,3,['Scores are from original full videos. The shorter episode ends and its frame is held.','The corrected scene continues, but instruction completion remains partial.'],[rows['IF022'][a]['video_seconds'] for a in ['L','C']]),
          lambda:v.comparison_segment('memory','Separate experiment: recover the earlier view','GC007 / scripted 31.8-second revisit, including a 24-second hold.',memory/'stock.mp4',memory/'retrieval.mp4',('STOCK / DINO 0.574','ARCHIVED MEMORY / DINO 0.746'),14,2.5,['Full episodes at 2.5×; final frames held. Same frozen model; historical KV retrieval.','DINO is a visual similarity diagnostic, not a PlayWorld score.'],[31.8125,31.8125]),lambda:v.still_segment(close,12,'08')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:segments=list(pool.map(lambda f:f(),jobs))
    quad,sources=segments[2];segments[2]=quad
    listing=v.WORK/'concat.txt';listing.write_text(''.join(f"file '{p}'\n" for p in segments))
    target=OUT/'worldsync-academic-2min.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(target)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(target)],text=True))
    assert len(probe['streams'])==1
    stream=probe['streams'][0];assert stream['codec_type']=='video' and int(stream['nb_read_frames'])==3600 and float(stream['duration'])==120
    receipt={'video':str(target.relative_to(v.ROOT)),'sha256':v.digest(target),'bytes':target.stat().st_size,'duration_seconds':120,'frames':3600,'width':1280,'height':720,'audio':'none: silent for live presenting','quadrant_sources':sources,'timeline_seconds':{'problem':[0,10],'benchmark':[10,22],'four_failure_videos':[22,42],'architecture':[42,62],'results':[62,74],'IF_comparison':[74,94],'memory_comparison':[94,108],'limitations_future':[108,120]}}
    (OUT/'academic-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    # Static backup uses actual rendered frames, including video panels.
    import cv2
    import shutil
    cap = cv2.VideoCapture(str(target))
    pages = []
    for second in [5,15,30,50,65,80,100,115]:
        cap.set(cv2.CAP_PROP_POS_MSEC, second*1000)
        ok, frame = cap.read()
        assert ok
        pages.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    pages[0].save(OUT/'worldsync-academic-slides.pdf', save_all=True,
                  append_images=pages[1:], resolution=120)
    pages[2].save(OUT/'four-dimensions.jpg')
    shutil.copy2(arch, OUT/'architecture.png')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
