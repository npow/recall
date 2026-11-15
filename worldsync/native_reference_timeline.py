"""Quarter-second camera schedule for the public action DSL, independent of render chunks."""
import math
import re
import numpy as np

KEYS={'KeyW':'W','KeyS':'S','KeyA':'A','KeyD':'D','ArrowLeft':'LEFT','ArrowRight':'RIGHT','ArrowUp':'UP','ArrowDown':'DOWN'}
TURN={'LEFT','RIGHT','UP','DOWN'}


class NativeReferenceTimeline:
    def __init__(self, actions, objective, nominal_turn_rate=50, phase_angles=None):
        if not isinstance(actions, list) or not actions:
            raise ValueError('A nonempty canonical action plan is required')
        if isinstance(nominal_turn_rate, bool) or not math.isfinite(nominal_turn_rate) or nominal_turn_rate <= 0:
            raise ValueError('Turn rate must be finite and positive')
        self.phases=[];self.phase=0;self.nominal_turn_rate=nominal_turn_rate
        elapsed=previous=0
        for raw in actions:
            if (raw.get('type') not in {'hold', 'wait'} or type(raw.get('duration_ms')) is not int
                    or raw['duration_ms'] <= 0
                    or (raw['type'] == 'wait' and raw.get('keys'))
                    or (raw['type'] == 'hold' and (len(raw.get('keys', [])) != 1 or raw['keys'][0] not in KEYS))):
                raise ValueError('Invalid canonical action')
            duration=raw['duration_ms'];elapsed+=duration;end=round(elapsed/250)
            steps=end-previous;previous=end
            if duration>0 and steps==0:raise ValueError('Reference action is shorter than the native250ms control interval')
            action='WAIT' if raw['type']=='wait' else KEYS[raw['keys'][0]]
            self.phases.append({'source':dict(raw),'action':action,'steps':steps,'used':0,'extension':0,'checks':0,'corrections':[],'angle':None})
        angles=[float(v) for v in re.findall(r'(\d+(?:\.\d+)?)\s*(?:-?\s*degrees?|°)',objective,re.I)]
        turns=[p for p in self.phases if p['action'] in TURN]
        if len(angles)==len(turns):
            for p,a in zip(turns,angles):p['angle']=a
        elif len(angles)==1 and len(turns)==1:turns[0]['angle']=angles[0]
        elif (len(angles)==1 and len(turns)==2 and {p['action'] for p in turns} in [{'LEFT','RIGHT'},{'UP','DOWN'}]
              and any(word in objective.lower() for word in ['back','starting','original','return'])):
            for p in turns:p['angle']=angles[0]
        if phase_angles is not None:
            expected={i for i,p in enumerate(self.phases) if p['action'] in TURN}
            if set(phase_angles) != expected:
                raise ValueError('Explicit bindings must cover exactly the turn phases')
            for index, angle in phase_angles.items():
                if isinstance(angle,bool) or not isinstance(angle,(int,float)) or not math.isfinite(angle) or angle <= 0:
                    raise ValueError('Invalid explicit turn angle')
                self.phases[index]['angle']=angle
        self.quantization={'interval_ms':250,'method':'Nearest cumulative boundary; explicit degree targets divided across their rounded phase steps',
                           'reference_total_ms':elapsed,'scheduled_total_ms':previous*250,
                           'nominal_turn_rate_for_unspecified_angles':nominal_turn_rate,
                           'explicit_angle_phases':sum(p['angle'] is not None for p in turns),
                           'unresolved_angle_phases':sum(p['angle'] is None for p in turns)}
        self.quantization['angle_binding']='Text-number heuristic; not validated semantic grounding'
        if phase_angles is not None:
            self.quantization['angle_binding']='Explicit task-fingerprinted phase bindings from released objective'
        self.quantization['native_translation_normalization_validated']=False
        if angles and any(p['angle'] is None for p in turns):
            raise ValueError('Explicit angle targets cannot be unambiguously bound to reference turns')

    def current(self):
        while self.phase<len(self.phases):
            p=self.phases[self.phase]
            if p.get('release_requested') and p['used'] >= 4:
                p['used']=p['steps']+p['extension']
            if p['used']<p['steps']+p['extension']:return p
            corrections=[]
            for direction in p['corrections']:
                corrections.append({'source':{'type':'hold','keys':[{'LEFT':'ArrowLeft','RIGHT':'ArrowRight','UP':'ArrowUp','DOWN':'ArrowDown'}[direction]],'duration_ms':1000,'raw':'released correction'},'action':direction,'steps':4,'used':0,'extension':0,'checks':3,'corrections':[],'angle':None})
            p['corrections']=[]
            self.phases[self.phase+1:self.phase+1]=corrections
            self.phase+=1
        return None

    def check_due(self):
        p=self.current()
        if p is None or p['action']=='WAIT' or p['action'] in TURN or p['checks']>=3:return False
        spacing=max(1500,p['source']['duration_ms']/3)
        return p['used']*250>=p['checks']*spacing

    def apply_decision(self, decision):
        direction=decision.get('adaptive_direction')
        extension=decision.get('extend_ms',0)
        if (type(decision.get('release_now')) is not bool
                or (direction is not None and direction not in KEYS)
                or (direction is not None and KEYS[direction] not in TURN)
                or type(extension) is not int or not 0 <= extension <= 5000):
            raise ValueError('Invalid released hold decision')
        p=self.current()
        if p is None:raise ValueError('No active reference phase')
        p['checks']+=1
        direction=KEYS.get(direction)
        if direction in TURN and direction not in p['corrections'] and len(p['corrections'])<2:p['corrections'].append(direction)
        if decision['release_now']:
            p['release_requested']=True
            if p['used']*250>=1000:p['used']=p['steps']+p['extension']
        else:p['extension']+=min(round(extension/250),40-p['extension'])

    def take(self, count=4):
        result=[]
        for _ in range(count):
            p=self.current()
            if p is None:result.append({'action':'WAIT','turn_degrees':0});continue
            # Preserve each phase's intended displacement despite clock rounding.
            duration_scale=p['source']['duration_ms']/(250*p['steps'])
            angle=(p['angle']/p['steps'] if p['angle'] is not None else self.nominal_turn_rate/4*duration_scale) if p['action'] in TURN else 0
            result.append({'action':p['action'],'turn_degrees':angle,
                           'translation_scale':duration_scale});p['used']+=1
        return result


def apply_camera_steps(camera, steps):
    expected=3 if not camera.poses else 4
    if len(steps)!=expected:raise ValueError('Schedule must fill exactly the native latent motion intervals')
    for item in steps:
        angle=item.get('turn_degrees')
        distance=item.get('translation_scale',1)
        if (item.get('action') not in set(KEYS.values()) | {'WAIT'} or isinstance(angle,bool)
                or not isinstance(angle,(int,float)) or not math.isfinite(angle) or angle < 0
                or isinstance(distance,bool) or not isinstance(distance,(int,float))
                or not math.isfinite(distance) or distance < 0):
            raise ValueError('Invalid native camera step')
    if not camera.poses:camera.poses.append(np.eye(4))
    for item in steps:
        action=item['action'];step=np.eye(4)
        translations={'W':[0,0,1],'S':[0,0,-1],'A':[-1,0,0],'D':[1,0,0]}
        if action in translations:step[:3,3]=np.asarray(translations[action])*camera.speed*item.get('translation_scale',1)
        elif action in TURN:
            angle=math.radians(item['turn_degrees'])*(1 if action in ['RIGHT','UP'] else -1);c,s=math.cos(angle),math.sin(angle)
            step[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]] if action in ['LEFT','RIGHT'] else [[1,0,0],[0,c,-s],[0,s,c]]
            if action in ['LEFT','RIGHT']:camera.yaw_commanded+=angle
        elif action!='WAIT':raise ValueError('Unsupported native action in timeline')
        camera.poses.append(camera.poses[-1]@step)
    return np.stack(camera.poses)
