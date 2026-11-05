#!/usr/bin/env python3
"""Conservative repair installer for RECORDING_PATCH_V1. Python stdlib only."""
from __future__ import annotations
import argparse, ast, os, re, sys, time, traceback
from pathlib import Path

REPAIR='RECORDING_REPAIR_V2'
TARGETS=('app.py','transcription_backend.py','web/index.html')

APP_BLOCK=r'''
# RECORDING_PATCH_V1: bounded, explicit-save PCM recording
# RECORDING_REPAIR_V2
import os as _rec_os
import threading as _rec_threading
import wave as _rec_wave
import uuid as _rec_uuid
import traceback as _rec_traceback
from pathlib import Path as _rec_Path
import numpy as _rec_np
from fastapi.responses import JSONResponse as _rec_JSONResponse
from starlette.concurrency import run_in_threadpool as _rec_run_in_threadpool

class _SessionRecording:
    """Bounded PCM16 recorder. All capture failures are intentionally contained."""
    def __init__(self):
        self.max_bytes=64*1024*1024
        self._lock=_rec_threading.Lock(); self._data=bytearray(); self.sample_rate=0
        self.saved=False; self.saving=False; self.capture_active=False; self.limit_reached=False
        self.generation=0; self.filename=None; self.fault=None
    def start(self, sample_rate):
        try: sr=int(sample_rate)
        except Exception: return False
        if sr < 8000 or sr > 384000: return False
        with self._lock:
            if self.saving: return False
            self._data=bytearray(); self.sample_rate=sr; self.saved=False; self.limit_reached=False
            self.capture_active=True; self.generation+=1; self.filename=None; self.fault=None
            return True
    def stop(self):
        with self._lock: self.capture_active=False
    def append(self, frame):
        with self._lock:
            if not self.capture_active or self.saved or self.limit_reached: return
            remaining=self.max_bytes-len(self._data)
            if remaining<=0: self.limit_reached=True; return
            a=_rec_np.asarray(frame)
            if a.ndim>1: a=a[:,0]
            a=_rec_np.nan_to_num(a,nan=0.0,posinf=1.0,neginf=-1.0)
            pcm=_rec_np.rint(_rec_np.clip(a.astype(_rec_np.float32,copy=False),-1,1)*32767).astype('<i2',copy=False).tobytes()
            take=min(len(pcm),remaining); take-=take%2; self._data.extend(pcm[:take])
            if take<len(pcm) or len(self._data)>=self.max_bytes: self.limit_reached=True
    def append_safe(self, frame):
        try: self.append(frame)
        except Exception:
            # Capture must never take down realtime transcription.
            with self._lock: self.fault='append_failed'
            try: print('recording append failed:\n'+_rec_traceback.format_exc(), file=sys.stderr)
            except Exception: pass
    def status(self):
        with self._lock:
            # A stopped backend is authoritative; do not leave stale active state after stop.
            if not STATE.running: self.capture_active=False
            n=len(self._data); active=self.capture_active
            return {'bytes':n,'seconds':n/(2*self.sample_rate) if self.sample_rate else 0,
             'max_bytes':self.max_bytes,'limit_reached':self.limit_reached,'saved':self.saved,
             'saving':self.saving,'capture_active':active,'available':bool(n and not self.saved and not self.saving and not STATE.running),
             'generation':self.generation,'filename':self.filename,'fault':self.fault}
    def save(self):
        req=_rec_uuid.uuid4().hex
        with self._lock:
            if not STATE.running: self.capture_active=False
            if self.saved: return {'success':True,'filename':self.filename,'request_id':req,'generation':self.generation},200
            if STATE.running: return {'success':False,'error':'recording_active','request_id':req},409
            if self.saving: return {'success':False,'error':'save_in_progress','request_id':req},409
            if not self._data or not self.sample_rate: return {'success':False,'error':'no_recording','request_id':req},409
            self.saving=True; data=bytes(self._data); sr=int(self.sample_rate); generation=self.generation
        path=None
        try:
            directory=_rec_Path(_rec_os.environ.get('APP_RECORDINGS_DIR','recordings')); directory.mkdir(parents=True,exist_ok=True)
            name='recording-'+_rec_uuid.uuid4().hex+'.wav'; path=directory/name
            with open(path,'xb') as f:
                with _rec_wave.open(f,'wb') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(data)
            with self._lock: self.saved=True; self.saving=False; self.filename=name
            return {'success':True,'filename':name,'request_id':req,'generation':generation},200
        except Exception:
            try: print('recording save failed request_id='+req+'\n'+_rec_traceback.format_exc(),file=sys.stderr)
            except Exception: pass
            if path is not None:
                try: path.unlink(missing_ok=True)
                except Exception: pass
            with self._lock: self.saving=False; self.fault='save_failed'
            return {'success':False,'error':'save_failed','request_id':req},500
RECORDING=_SessionRecording()
@app.get('/api/recording/status')
async def recording_status(): return _rec_JSONResponse(RECORDING.status())
def _recording_save_sync(): return RECORDING.save()
@app.post('/api/recording/save')
async def recording_save():
    payload,code=await _rec_run_in_threadpool(_recording_save_sync)
    return _rec_JSONResponse(payload,status_code=code)
'''

HTML_BLOCK=r'''
<!-- RECORDING_PATCH_V1 --><!-- RECORDING_REPAIR_V2 -->
<script>
(()=>{const stop=document.getElementById('btn-stop');if(!stop)return;const save=document.createElement('button'), note=document.createElement('span');save.id='btn-save-recording';save.type='button';note.id='recording-save-status';note.style.marginLeft='8px';save.className=stop.className;stop.insertAdjacentElement('afterend',save);save.insertAdjacentElement('afterend',note);let busy=false,saving=false,lastGen=null,last=null,err='';
function de(){let x=(document.getElementById('lang-select')||{}).value||document.documentElement.lang||localStorage.getItem('ui_lang')||localStorage.getItem('lang')||'en';return /^de/i.test(x)}function tx(){let g=de(),s=last||{};save.textContent=g?'Aufnahme speichern':'Save recording';let a=g?'Aufnahme':'Recording',t=`${a}: ${Number(s.seconds||0).toFixed(1)} s`;if(s.limit_reached)t+=g?' – RAM-Limit erreicht; STT läuft weiter.':' – RAM limit reached; STT continues.';if(s.saved)t+=` – ${(g?'gespeichert':'saved')}${s.filename?': '+s.filename:''}.`;else t+=g?' – RAM-Audio geht bei Neustart verloren.':' – RAM audio is lost on restart.';if(err)t+=' – '+err;note.textContent=t;save.disabled=!!(saving||!s.available||s.saved||s.saving||s.capture_active)}
function message(j,code){let g=de(),e=(j&&j.error)||'network_error',id=j&&j.request_id;let m={recording_active:g?'Aufnahme läuft noch.':'Recording is still active.',save_in_progress:g?'Speichern läuft bereits.':'Save already in progress.',no_recording:g?'Keine Aufnahme verfügbar.':'No recording available.',save_failed:g?'Speichern fehlgeschlagen.':'Save failed.',network_error:g?'Netzwerkfehler.':'Network error.'}[e]||(g?'Speicherfehler':'Save error');return m+' ['+e+(code?' / '+code:'')+(id?' / '+id:'')+']'}
async function poll(){if(busy)return;busy=true;try{let r=await fetch('/api/recording/status',{credentials:'same-origin'}),j=await r.json();if(r.ok){if(lastGen!==null&&j.generation!==lastGen)err='';lastGen=j.generation;last=j;tx()}}catch(_){err=message(null,0);tx()}finally{busy=false}}
save.addEventListener('click',async()=>{if(saving)return;saving=true;err='';tx();try{let r=await fetch('/api/recording/save',{method:'POST',credentials:'same-origin'}),j;try{j=await r.json()}catch(_){j=null}if(!r.ok||!j||!j.success)err=message(j,r.status);else {last=Object.assign(last||{},j,{saved:true});err=''}}catch(_){err=message(null,0)}finally{saving=false;await poll();tx()}});document.addEventListener('change',tx);new MutationObserver(tx).observe(document.documentElement,{attributes:true,attributeFilter:['lang']});window.addEventListener('storage',tx);poll();setInterval(poll,2000)})();
</script>
'''

def span(lines,n): return sum(map(len,lines[:n.lineno-1])),sum(map(len,lines[:n.end_lineno]))
def indent(text,pos):
 p=text.rfind('\n',0,pos)+1; e=text.find('\n',pos); e=len(text) if e<0 else e
 line=text[p:e]; return line[:len(line)-len(line.lstrip())]
def replace_spans(text,changes):
 for a,b,v in sorted(changes,reverse=True): text=text[:a]+v+text[b:]
 return text

def patch_app(t):
 lines=t.splitlines(keepends=True); tree=ast.parse(t); top=tree.body
 marker='# RECORDING_PATCH_V1: bounded, explicit-save PCM recording'
 if t.count(marker)!=1: raise ValueError('app recording marker must be unique')
 end=next((n for n in top if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='recording_save'),None)
 if not end: raise ValueError('recording_save endpoint not found')
 a=t.index(marker); a=t.rfind('\n',0,a)+1; b=span(lines,end)[1]
 t=replace_spans(t,[(a,b,APP_BLOCK+'\n')])
 if 'RECORDING.append(frame)' not in t: raise ValueError('original recording append hook missing')
 t=t.replace('RECORDING.append(frame)','RECORDING.append_safe(frame)',1)
 compile(t,'app.py','exec'); return t

def patch_backend(t):
 tree=ast.parse(t); lines=t.splitlines(keepends=True); cls=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RealtimeTranscriber'),None)
 if not cls: raise ValueError('RealtimeTranscriber missing')
 worker=next((n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_worker'),None); start=next((n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='start'),None)
 if not worker or not start: raise ValueError('worker/start missing')
 chosen=[]
 for n in ast.walk(worker):
  if isinstance(n,ast.If) and 'candidate_txt' in ast.unparse(n.test) and 'not candidate_txt' in ast.unparse(n.test) and any(isinstance(x,ast.Await) and isinstance(x.value,ast.Call) and getattr(x.value.func,'attr',None)=='_decode_partial' for x in ast.walk(n)): chosen.append(n)
 if len(chosen)!=1: raise ValueError('fallback block ambiguous or absent')
 n=chosen[0]; a,b=span(lines,n); ind=indent(t,a); cond=ast.get_source_segment(t,n.test)
 block=f"{ind}# RECORDING_REPAIR_V2: bounded fallback; errors are observable\n{ind}if {cond}:\n{ind}    try:\n{ind}        import sys\n{ind}        pcm = (np.frombuffer(self._record_bytes.getvalue(), dtype=np.float32).copy() if self._record_bytes is not None else np.empty(0, dtype=np.float32))\n{ind}        if pcm.size:\n{ind}            fb_txt, _ = await self._decode_partial(pcm, finalize=True)\n{ind}            fb_txt = clean_transcript(fb_txt)\n{ind}            if fb_txt:\n{ind}                candidate_txt = fb_txt\n{ind}                fallback_used = True\n{ind}            else:\n{ind}                print('realtime fallback returned empty transcript', file=sys.stderr)\n{ind}    except Exception:\n{ind}        import traceback as _rt_traceback\n{ind}        print('realtime fallback failed:\\n' + _rt_traceback.format_exc(), file=sys.stderr)\n"
 # callback follows unique task assignment in start
 assigns=[x for x in ast.walk(start) if isinstance(x,ast.Assign) and any(isinstance(z,ast.Attribute) and z.attr=='_task' for z in x.targets)]
 if len(assigns)!=1: raise ValueError('_task assignment ambiguous')
 task_start,e=span(lines,assigns[0]); taskind=indent(t,task_start)
 callback=f"\n{taskind}self._task.add_done_callback(_recording_repair_log_task_exception)\n"
 # helper at module end avoids imports / awaits in callback
 helper="\n\ndef _recording_repair_log_task_exception(task):\n    import sys\n    try:\n        if not task.cancelled():\n            exc = task.exception()\n            if exc is not None:\n                import traceback\n                print('realtime worker crashed:\\n' + ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)), file=sys.stderr)\n    except Exception:\n        pass\n"
 t=replace_spans(t,[(a,b,block),(e,e,callback)])+helper
 compile(t,'transcription_backend.py','exec');return t

def patch_html(t):
 marker='<!-- RECORDING_PATCH_V1 -->'; a=t.find(marker)
 if a<0 or t.find(marker,a+1)>=0: raise ValueError('HTML marker must be unique')
 m=re.search(r'<script\b[^>]*>.*?</script>',t[a:],re.S|re.I)
 if not m: raise ValueError('recording script missing')
 b=a+m.end(); result=t[:a]+HTML_BLOCK+t[b:]; return result

def install(root):
 ps=[root/x for x in TARGETS]
 if any(not p.is_file() for p in ps): raise ValueError('required exact targets absent: '+', '.join(str(p) for p in ps if not p.is_file()))
 old={p:p.read_bytes() for p in ps}; text={p:old[p].decode('utf-8') for p in ps}
 marks=[REPAIR in text[p] for p in ps]
 if all(marks): print('Already repaired; no files changed.');return
 if any(marks): raise ValueError('partial '+REPAIR+' markers; refusing to write')
 new={ps[0]:patch_app(text[ps[0]]),ps[1]:patch_backend(text[ps[1]]),ps[2]:patch_html(text[ps[2]])}
 for p,v in new.items():
  if REPAIR not in v: raise ValueError('repair marker absent '+str(p))
  if p.suffix=='.py': compile(v,str(p),'exec')
 stamp=time.strftime('%Y%m%d%H%M%S'); backups={}
 try:
  for p in ps:
   q=p.with_name(p.name+'.recording_repair.'+stamp+'.bak'); i=0
   while q.exists(): i+=1;q=p.with_name(p.name+f'.recording_repair.{stamp}.{i}.bak')
   q.write_bytes(old[p]);backups[p]=q
  for p,v in new.items():
   tmp=p.with_name(p.name+'.recording_repair.tmp');tmp.write_text(v,encoding='utf-8');os.replace(tmp,p)
 except Exception:
  for p,bak in backups.items():
   try: os.replace(bak,p); bak.write_bytes(old[p])
   except Exception: p.write_bytes(old[p])
  raise
 print('Repaired '+', '.join(TARGETS)+'; backups: '+', '.join(str(x) for x in backups.values()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('repo',nargs='?',default='.',type=Path);a=ap.parse_args();install(a.repo.resolve())
if __name__=='__main__':
 try: main()
 except Exception as e: print('Repair not applied: '+str(e),file=sys.stderr);sys.exit(1)
