#!/usr/bin/env python3
"""Conservative stdlib-only installer; it never imports or executes the target app."""
from __future__ import annotations
import argparse, ast, os, re, stat, sys, tempfile, time, uuid
from pathlib import Path
MARKER='RECORDING_PATCH_V1'; REPAIR='RECORDING_SAVE_REPAIR_V3'
COMMENT='# Leave empty to auto-detect language; LOWEND_ forces "de".'
def span(ls,n): return sum(map(len,ls[:n.lineno-1])),sum(map(len,ls[:n.end_lineno]))
def ind(s,p):
 a=s.rfind('\n',0,p)+1;b=s.find('\n',p);b=len(s) if b<0 else b; x=s[a:b];return x[:len(x)-len(x.lstrip())]
def repl(s,a,b,x): return s[:a]+x+s[b:]
def save_source(i):
 s="""def save(self):
    # Imports are local deliberately: no application startup behavior is changed.
    import builtins
    import os
    import wave
    import uuid
    import logging
    from pathlib import Path
    request_id = uuid.uuid4().hex
    path = None
    file_created = False
    claimed = False
    logger = logging.getLogger(__name__)
    try:
        with self._lock:
            if STATE.running:
                return {"success": False, "error": "recording_active", "request_id": request_id}, 409
            if self.saving:
                return {"success": False, "error": "save_in_progress", "request_id": request_id}, 409
            if self.saved:
                return {"success": True, "filename": getattr(self, "filename", None), "generation": getattr(self, "generation", None), "request_id": request_id}, 200
            if not self._data or not self.sample_rate:
                return {"success": False, "error": "no_recording", "request_id": request_id}, 409
            self.saving = True
            claimed = True
            data = bytes(self._data)
            sample_rate = int(self.sample_rate)
            generation = getattr(self, "generation", None)
        if sample_rate < 8000 or sample_rate > 384000:
            raise ValueError("invalid_sample_rate")
        if len(data) % 2:
            raise ValueError("odd_pcm16_bytes")
        directory = Path(os.environ.get("APP_RECORDINGS_DIR", "recordings")).expanduser()
        if not directory.is_absolute():
            directory = Path(__file__).resolve().parent / directory
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        name = "recording-" + uuid.uuid4().hex + ".wav"
        path = directory / name
        with builtins.open(path, "xb") as f:
            file_created = True
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(data)
        with self._lock:
            self.saved = True; self.saving = False; self.filename = name; self.fault = None
        logger.info("recording saved request_id=%s path=%s", request_id, str(path))
        return {"success": True, "filename": name, "generation": generation, "request_id": request_id}, 200
    except Exception as exc:
        try:
            logger.exception("recording save failed request_id=%s", request_id)
        except Exception:
            import traceback
            import sys
            try: traceback.print_exc(file=sys.stderr)
            except Exception: pass
        if file_created and path is not None:
            try: path.unlink(missing_ok=True)
            except Exception: pass
        if claimed:
            with self._lock:
                self.saving = False; self.fault = "save_failed"
        return {"success": False, "error": "save_failed", "error_type": type(exc).__name__, "request_id": request_id}, 500
"""
 return ''.join(i+x if x.strip() else x for x in s.splitlines(keepends=True))
def patch_app(text):
 if text.count(MARKER)!=1: raise ValueError('app.py must contain exactly one RECORDING_PATCH_V1 marker')
 if REPAIR in text:return text,False
 ls=text.splitlines(keepends=True);t=ast.parse(text,'app.py')
 cs=[x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='_SessionRecording']
 if len(cs)!=1:raise ValueError('required unique _SessionRecording class not found')
 c=cs[0]; ss=[x for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='save']; aa=[x for x in c.body if isinstance(x,ast.AsyncFunctionDef) and x.name=='save']
 if len(ss)!=1 or aa:raise ValueError('required unique non-async _SessionRecording.save not found')
 fields={x.attr for x in ast.walk(c) if isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id=='self'}
 if not {'_lock','_data','sample_rate','saving','saved'}<=fields:raise ValueError('_SessionRecording lacks required state fields')
 a,b=span(ls,ss[0]);text=repl(text,a,b,save_source(ind(text,a)))
 # Reparse after replacement, then scope the original append_safe stderr repair to this class only.
 t=ast.parse(text,'app.py');ls=text.splitlines(keepends=True);c=[x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='_SessionRecording'][0];a,b=span(ls,c)
 text=repl(text,a,b,text[a:b].replace('file=sys.stderr','file=_rec_sys.stderr'))
 t=ast.parse(text,'app.py')
 if not any(isinstance(x,ast.Import) and any(y.name=='sys' and y.asname=='_rec_sys' for y in x.names) for x in t.body):
  p=text.rfind('\n',0,text.index(MARKER))+1;text=repl(text,p,p,'import sys as _rec_sys\n')
 text+=("" if text.endswith('\n') else '\n')+'# '+REPAIR+'\n';compile(text,'app.py','exec');return text,True
def patch_env(text):
 nl='\r\n' if '\r\n' in text else '\n'; final=text.endswith('\n') or text.endswith('\r');out=[];changed=False
 for row in text.splitlines():
  m=re.match(r'^(\s*(?:export\s+)?APP_WHISPER_LANGUAGE\s*=\s*)(.*)$',row); bad=False
  if m:
   v=m.group(2).strip();bad=v.startswith('#') or (len(v)>=2 and v[0] in "\"'" and v[-1]==v[0] and v[1:-1].lstrip().startswith('#'))
  if bad:
   if not out or out[-1]!=COMMENT:out.append(COMMENT)
   out.append(m.group(1));changed=True
  else:out.append(row)
 r=nl.join(out)+(nl if final else '');return r,changed
def backup(p):
 while True:
  q=p.with_name(p.name+'.recording_save.'+time.strftime('%Y%m%d%H%M%S')+'.'+uuid.uuid4().hex+'.bak')
  if not q.exists():return q
def atomic(p,data,mode):
 fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',suffix='.recording-save.tmp',dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.chmod(tmp,stat.S_IMODE(mode));os.replace(tmp,p)
 except Exception:
  try:os.unlink(tmp)
  except OSError:pass
  raise
def install(root):
 root=root.resolve();app=root/'app.py';env=root/'.env'
 if not app.is_file():raise ValueError('required target app.py is absent')
 olda=app.read_bytes()
 try:newa,chga=patch_app(olda.decode('utf-8'))
 except UnicodeDecodeError as e:raise ValueError('app.py is not UTF-8') from e
 exists=env.exists()
 if exists and not env.is_file():raise ValueError('.env exists but is not regular')
 olde=env.read_bytes() if exists else b''
 try:newe,chge=patch_env(olde.decode('utf-8'))
 except UnicodeDecodeError as e:raise ValueError('.env is not UTF-8') from e
 compile(newa,str(app),'exec') # all staging/validation precedes backups/writes
 jobs=[]
 if chga:jobs.append((app,olda,newa.encode(),app.stat().st_mode))
 if chge or not exists:jobs.append((env,olde,newe.encode(),env.stat().st_mode if exists else 0o600))
 if not jobs:
  print('Already repaired; no files changed.');print('Restart the application to use recording-save behavior.');return
 bs=[];done=[]
 try:
  for p,o,n,m in jobs:
   q=backup(p);q.write_bytes(o);os.chmod(q,stat.S_IMODE(m));bs.append(q)
  for p,o,n,m in jobs:atomic(p,n,m);done.append((p,o,m))
 except Exception:
  for p,o,m in reversed(done):
   try:atomic(p,o,m)
   except Exception:pass
  raise
 print('Repair installed: '+', '.join(p.name for p,_,_,_ in jobs));print('Backups: '+', '.join(map(str,bs)))
 print('APP_WHISPER_LANGUAGE corrected to empty: automatic language detection is enabled.')
 if os.environ.get('APP_WHISPER_LANGUAGE','').lstrip().startswith('#'):print('Warning: current process APP_WHISPER_LANGUAGE begins with #; restart from a clean environment. This installer cannot change its parent environment.')
 print('Restart the application to apply the repair.')
def fixture():return '''# RECORDING_PATCH_V1
import threading
class _SessionRecording:
 def __init__(self):
  self._lock=threading.Lock();self._data=bytearray();self.sample_rate=16000;self.saving=False;self.saved=False;self.generation=1;self.filename=None;self.fault=None
 def save(self): return {},500
'''
def selftest():
 import shutil,wave
 from types import SimpleNamespace
 d=Path(tempfile.mkdtemp());oldenv=os.environ.get('APP_RECORDINGS_DIR')
 try:
  (d/'app.py').write_text(fixture());(d/'.env').write_text('APP_WHISPER_LANGUAGE="# bad"\nOTHER=secret\n');install(d);a=(d/'app.py').read_bytes();e=(d/'.env').read_bytes();install(d);assert a==(d/'app.py').read_bytes() and e==(d/'.env').read_bytes()
  src=a.decode();compile(src,str(d/'app.py'),'exec');ns={'__file__':str(d/'app.py'),'STATE':SimpleNamespace(running=False),'open':lambda *x:(_ for _ in ()).throw(RuntimeError())};exec(compile(src,str(d/'app.py'),'exec'),ns);r=ns['_SessionRecording']();r._data=bytearray(b'\0\0\1\0');os.environ['APP_RECORDINGS_DIR']=str(d/'r');x,c=r.save();assert c==200
  with wave.open(str(d/'r'/x['filename'])) as w:assert w.getnframes()==2
  assert r.save()[1]==200;r=ns['_SessionRecording']();r._data=bytearray(b'\0');assert r.save()[1]==500 and not r.saving and r._data;ns['STATE'].running=True;r=ns['_SessionRecording']();r._data=bytearray(b'\0\0');assert r.save()[1]==409
  ref=Path(__file__).with_name('repair_recording_patch.py')
  if ref.exists():
   t=ast.parse(ref.read_text());v=[n.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id=='APP_BLOCK' for z in n.targets)];
   if v:compile(ast.literal_eval(v[0]),'reference_APP_BLOCK.py','exec')
  print('Tests passed: installer transaction, dotenv normalization, AST patch, WAV save, failure reset, idempotence, running guard, and builtins.open isolation.');print('Untested: real application integration, audio hardware, and production filesystem permissions.')
 finally:
  if oldenv is None:os.environ.pop('APP_RECORDINGS_DIR',None)
  else:os.environ['APP_RECORDINGS_DIR']=oldenv
  shutil.rmtree(d,ignore_errors=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('repo',nargs='?',default='.',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args();selftest() if a.self_test else install(a.repo)
if __name__=='__main__':
 try:main()
 except Exception as e:print('Repair not applied: '+type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(1)
