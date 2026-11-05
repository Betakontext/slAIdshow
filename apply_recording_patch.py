#!/usr/bin/env python3
"""Apply the in-memory recording patch.  Python 3.12 stdlib only."""
from __future__ import annotations
import argparse, ast, re, shutil, sys
from pathlib import Path

MARK = "RECORDING_PATCH_V1"
EXCLUDED = {".git", ".venv", "venv", "env", "node_modules", "outputs", "recordings", "__pycache__"}

APP_BLOCK = r'''
# RECORDING_PATCH_V1: bounded, explicit-save PCM recording
import os as _rec_os
import threading as _rec_threading
import wave as _rec_wave
import uuid as _rec_uuid
from pathlib import Path as _rec_Path
import numpy as _rec_np
from fastapi.responses import JSONResponse as _rec_JSONResponse
from starlette.concurrency import run_in_threadpool as _rec_run_in_threadpool

class _SessionRecording:
    def __init__(self):
        try:
            _mib = int(_rec_os.environ.get("APP_RECORDING_MAX_MIB", "64"))
        except ValueError:
            _mib = 64
        self.max_bytes = min(256, max(1, _mib)) * 1024 * 1024
        self._lock = _rec_threading.Lock()
        self._data = bytearray()
        self.sample_rate = 0
        self.saved = False
        self.saving = False
        self.capture_active = False
        self.limit_reached = False
        self.generation = 0

    def start(self, sample_rate):
        with self._lock:
            if self.saving:
                return False
            self._data = bytearray()
            self.sample_rate = int(sample_rate)
            self.saved = False
            self.limit_reached = False
            self.capture_active = True
            self.generation += 1
            return True

    def stop(self):
        with self._lock:
            self.capture_active = False

    def append(self, frame):
        with self._lock:
            if not self.capture_active or self.saved or self.limit_reached:
                return
            remaining = self.max_bytes - len(self._data)
            if remaining <= 0:
                self.limit_reached = True
                return
            a = _rec_np.asarray(frame)
            if a.ndim > 1:
                a = a[:, 0]
            a = _rec_np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
            a = _rec_np.clip(a.astype(_rec_np.float32, copy=False), -1.0, 1.0)
            pcm = _rec_np.rint(a * 32767.0).astype("<i2", copy=False).tobytes()
            take = min(len(pcm), remaining)
            take -= take % 2
            self._data.extend(pcm[:take])
            if take < len(pcm) or len(self._data) >= self.max_bytes:
                self.limit_reached = True

    def status(self):
        with self._lock:
            n = len(self._data)
            active = self.capture_active
            return {"bytes": n, "seconds": (n / (2 * self.sample_rate) if self.sample_rate else 0),
                    "max_bytes": self.max_bytes, "limit_reached": self.limit_reached,
                    "saved": self.saved, "saving": self.saving, "capture_active": active,
                    "available": bool(n and not self.saved and not self.saving and not active and not STATE.running)}

    def save(self):
        with self._lock:
            if STATE.running or self.capture_active:
                return {"success": False, "error": "recording_active"}, 409
            if self.saving:
                return {"success": False, "error": "save_in_progress"}, 409
            if self.saved:
                return {"success": False, "error": "already_saved"}, 409
            if not self._data or not self.sample_rate:
                return {"success": False, "error": "no_recording"}, 409
            self.saving = True
            view = memoryview(self._data)
            sr = self.sample_rate
        path = None
        try:
            directory = _rec_Path(_rec_os.environ.get("APP_RECORDINGS_DIR", "recordings"))
            directory.mkdir(parents=True, exist_ok=True)
            name = "recording-" + _rec_uuid.uuid4().hex + ".wav"
            path = directory / name
            # Create exclusively first; wave itself requires its normal write mode.
            with open(path, "xb") as _rec_file:
                with _rec_wave.open(_rec_file, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                    wf.writeframesraw(view)
            view.release()
            with self._lock:
                self.saved = True; self.saving = False
            return {"success": True, "filename": name}, 200
        except Exception:
            try:
                view.release()
            except Exception:
                pass
            if path is not None:
                try: path.unlink(missing_ok=True)
                except Exception: pass
            with self._lock: self.saving = False
            return {"success": False, "error": "save_failed"}, 500

RECORDING = _SessionRecording()

@app.get("/api/recording/status")
async def recording_status():
    return _rec_JSONResponse(RECORDING.status())

def _recording_save_sync():
    return RECORDING.save()

@app.post("/api/recording/save")
async def recording_save():
    payload, code = await _rec_run_in_threadpool(_recording_save_sync)
    return _rec_JSONResponse(payload, status_code=code)
'''

HTML_BLOCK = r'''
<!-- RECORDING_PATCH_V1 -->
<script>
(() => {
  const stop = document.getElementById('btn-stop'); if (!stop) return;
  const save = document.createElement('button'); save.id='btn-save-recording'; save.type='button'; save.textContent='Save Recording'; save.disabled=true;
  const status = document.createElement('span'); status.id='recording-save-status'; status.style.marginLeft='8px';
  stop.insertAdjacentElement('afterend', save); save.insertAdjacentElement('afterend', status);
  let busy=false, error='';
  function show(s) {
    let t = `Aufnahme: ${Number(s.seconds||0).toFixed(1)} s`;
    if (s.limit_reached) t += ' – Speicherlimit erreicht; Aufnahme gestoppt, STT läuft weiter.';
    if (s.saved) { t += ' – gespeichert.'; error=''; }
    if (!s.saved) t += ' – Beim neuen Start oder beim Herunterfahren geht RAM-Audio verloren.';
    if (error) t += ' – ' + error;
    status.textContent=t;
    save.disabled=!(s.available && !s.saved && !s.saving && !s.capture_active);
  }
  async function poll() { if(busy) return; busy=true; try { const r=await fetch('/api/recording/status',{credentials:'same-origin'}); if(r.ok) show(await r.json()); } catch(_) { } finally {busy=false;} }
  save.addEventListener('click', async () => { error=''; save.disabled=true; try { const r=await fetch('/api/recording/save',{method:'POST',credentials:'same-origin'}); const j=await r.json(); if(!r.ok || !j.success) error='Speichern fehlgeschlagen.'; } catch(_) { error='Speichern fehlgeschlagen.'; } await poll(); });
  poll(); setInterval(poll, 2000);
})();
</script>
'''

def files(root):
    """Yield files below root, excluding named relative directory components."""
    for p in root.rglob('*'):
        try:
            parts = p.relative_to(root).parts
        except ValueError:
            continue
        if any(part in EXCLUDED for part in parts) or not p.is_file():
            continue
        yield p


def unique_named(root, name):
    got = [p for p in files(root) if p.name == name]
    if len(got) != 1:
        raise ValueError(f"expected exactly one {name}, found {len(got)}")
    return got[0]


def span(lines, node):
    return (sum(map(len, lines[:node.lineno - 1])),
            sum(map(len, lines[:node.end_lineno])))


def indent_of(text, at):
    """Return indentation of the complete physical source line containing at."""
    line_start = text.rfind('\n', 0, at) + 1
    line_end = text.find('\n', at)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    return line[:len(line) - len(line.lstrip(' \t'))]


def _stmt_is_record_path_assignment(node):
    targets = []
    if isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.Assign):
        targets = node.targets
    return any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
               and t.value.id == 'self' and t.attr == '_record_path' for t in targets)


def _is_flush_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'
            and node.func.attr == '_flush_recording_wav')


def _contains_flush_call(node):
    return any(_is_flush_call(n) for n in ast.walk(node))


def _contains_assigned_flush(node):
    for n in ast.walk(node):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and _contains_flush_call(n):
            return True
    return False


def _is_self_config_attr(node, attr):
    """True only for self.config.<attr>, not an unrelated same-named variable."""
    return (isinstance(node, ast.Attribute) and node.attr == attr
            and isinstance(node.value, ast.Attribute) and node.value.attr == 'config'
            and isinstance(node.value.value, ast.Name) and node.value.value.id == 'self')


def _has_config_fallback_attr(node):
    return any(_is_self_config_attr(n, name)
               for n in ast.walk(node) for name in ('rt_fallback_record', 'rt_fallback_dir'))


def _is_direct_mkdir_stmt(node):
    """A direct suite statement which calls mkdir, rather than an enclosing If."""
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'mkdir')


def _is_mkdir_fallback_if(node):
    # Deliberately require the mkdir as a direct body statement.  This prevents
    # removing a broader conditional which merely encloses a fallback setup.
    return (isinstance(node, ast.If) and _has_config_fallback_attr(node.test)
            and any(_is_direct_mkdir_stmt(stmt) for stmt in node.body))


def _is_record_write_expr(node):
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    f = call.func
    if not (isinstance(f, ast.Attribute) and f.attr == 'write' and isinstance(f.value, ast.Attribute)
            and isinstance(f.value.value, ast.Name) and f.value.value.id == 'self'
            and f.value.attr == '_record_bytes'):
        return False
    # Exact enough to avoid bounding some unrelated BytesIO write.
    return bool(call.args) and 'item.astype(np.float32, copy=False).tobytes()' == ast.unparse(call.args[0])


def _has_not_candidate_txt(test):
    return any(isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)
               and isinstance(n.operand, ast.Name) and n.operand.id == 'candidate_txt'
               for n in ast.walk(test))


def _is_transcribe_file_for_fallback_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'transcribe_file_for_fallback')


def _has_fallback_calls(if_node):
    # Search only its body (not test/orelse or enclosing source) for real calls.
    body_nodes = [n for stmt in if_node.body for n in ast.walk(stmt)]
    return (any(_is_flush_call(n) for n in body_nodes)
            and any(_is_transcribe_file_for_fallback_call(n) for n in body_nodes))


def _if_diagnostic(nodes):
    if not nodes:
        return 'none'
    return '; '.join('line %s: %s' % (n.lineno, ast.unparse(n.test)) for n in nodes)

def _replace_spans(text, changes):
    """Apply verified, non-overlapping source ranges from right to left."""
    ordered = sorted(changes, key=lambda c: (c[0], c[1]))
    previous_end = -1
    for start, end, _ in ordered:
        if start < previous_end or end < start:
            raise ValueError('overlapping or invalid AST edits')
        previous_end = max(previous_end, end)
    for start, end, replacement in reversed(ordered):
        text = text[:start] + replacement + text[end:]
    return text


def patch_trans(text):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RealtimeTranscriber']
    if len(classes) != 1:
        raise ValueError('RealtimeTranscriber must be unique')
    cls = classes[0]
    methods = {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required = {'__init__', 'start', 'stop', 'feed', '_worker', '_decode_partial', '_flush_recording_wav'}
    if not required <= methods.keys():
        raise ValueError('required transcriber methods missing')

    # Restrict discovery to _worker.  The desired guard is compound in current
    # sources, so `not candidate_txt` may occur anywhere below its test AST.
    worker = methods['_worker']
    candidate_tests = [n for n in ast.walk(worker) if isinstance(n, ast.If) and _has_not_candidate_txt(n.test)]
    candidates = [n for n in candidate_tests if _has_fallback_calls(n)]
    if not candidates:
        raise ValueError('no fallback If in _worker has `not candidate_txt` plus both required calls; '
                         'candidate tests: ' + _if_diagnostic(candidate_tests))
    # In a nested chain the innermost matching If is the only safe replacement.
    def contains(outer, inner):
        os, oe = span(lines, outer); ins, ine = span(lines, inner)
        return outer is not inner and os <= ins and ine <= oe
    innermost = [n for n in candidates if not any(contains(n, other) for other in candidates)]
    if len(innermost) != 1:
        raise ValueError('fallback If is ambiguous in _worker; matching tests: '
                         + _if_diagnostic(candidates)
                         + '; innermost tests: ' + _if_diagnostic(innermost))
    fallback = innermost[0]
    fs, fe = span(lines, fallback)
    f_indent = indent_of(text, fs)
    # Preserve the complete source condition, including all compound guards.
    condition = ast.get_source_segment(text, fallback.test)
    if not condition:
        raise ValueError('could not recover fallback condition source')
    fallback_replacement = (
        f'{f_indent}# In-memory fallback: decode the bounded realtime buffer\n'
        f'{f_indent}if {condition}:\n'
        f'{f_indent}    with contextlib.suppress(Exception):\n'
        f'{f_indent}        pcm = (np.frombuffer(self._record_bytes.getvalue(), dtype=np.float32)\n'
        f'{f_indent}               if self._record_bytes is not None else np.empty(0, dtype=np.float32))\n'
        f'{f_indent}        if pcm.size:\n'
        f'{f_indent}            fb_txt, _ = await self._decode_partial(pcm, finalize=True)\n'
        f'{f_indent}            fb_txt = clean_transcript(fb_txt)\n'
        f'{f_indent}            if fb_txt:\n'
        f'{f_indent}                candidate_txt = fb_txt\n'
        f'{f_indent}                fallback_used = True\n')

    removals = [methods['_flush_recording_wav']]
    removals += [n for n in ast.walk(cls) if _stmt_is_record_path_assignment(n)]
    # Only the smallest fallback mkdir If: the predicate itself requires a
    # direct mkdir statement and a self.config fallback attribute in its test.
    removals += [n for n in ast.walk(cls) if _is_mkdir_fallback_if(n)]
    # A shutdown suppress/with is removable only when its flush call is not assigned.
    removals += [n for n in ast.walk(cls) if isinstance(n, (ast.With, ast.AsyncWith))
                 and _contains_flush_call(n) and not _contains_assigned_flush(n)]

    # The fallback replacement owns its interior.  An enclosing removal would
    # overlap it and is unsafe; fail before any write instead of guessing.
    filtered = []
    for n in removals:
        s, e = span(lines, n)
        if fs <= s and e <= fe:
            continue
        if s <= fs and fe <= e:
            raise ValueError('legacy removal encloses selected fallback If at line %d' % fallback.lineno)
        filtered.append(n)
    # Deduplicate and retain outermost complete statements only.
    unique = {}
    for n in filtered:
        unique[span(lines, n)] = n
    removal_spans = sorted(unique)
    outermost = []
    for s, e in removal_spans:
        if not any((os < s or oe > e) and os <= s and e <= oe for os, oe in removal_spans):
            outermost.append((s, e, unique[(s, e)]))
    # Distinct non-contained spans must never overlap.
    for (_, e1, _), (s2, _, _) in zip(outermost, outermost[1:]):
        if e1 > s2:
            raise ValueError('overlapping legacy removal statements')

    write_nodes = [n for n in ast.walk(worker) if _is_record_write_expr(n)]
    if len(write_nodes) != 1:
        raise ValueError('record BytesIO write in _worker must be unique')
    write_node = write_nodes[0]
    ws, we = span(lines, write_node)
    if fs <= ws and we <= fe:
        raise ValueError('record BytesIO write lies inside selected fallback If')
    if any(s <= ws and we <= e for s, e, _ in outermost):
        raise ValueError('legacy removal would delete record BytesIO write')
    write_indent = indent_of(text, ws)
    bound = (f'\n{write_indent}if self._record_bytes.tell() > sr * 30 * 4:\n'
             f'{write_indent}    self._record_bytes.seek(-sr * 30 * 4, io.SEEK_END)\n'
             f'{write_indent}    self._record_bytes = io.BytesIO(self._record_bytes.read())\n')

    changes = [(fs, fe, fallback_replacement), (we, we, bound)]
    for s, e, n in outermost:
        parent_empty = any(isinstance(getattr(parent, field, None), list)
                           and len(getattr(parent, field)) == 1 and getattr(parent, field)[0] is n
                           for parent in ast.walk(cls) for field in ('body', 'orelse', 'finalbody'))
        replacement = (indent_of(text, s) + 'pass  # recording patch: removed legacy disk fallback\n'
                       if parent_empty else '')
        changes.append((s, e, replacement))
    result = _replace_spans(text, changes)  # validates all edits before any caller writes
    result = result.rstrip() + '\n\n# RECORDING_PATCH_V1: in-memory realtime fallback\n'
    try:
        compile(result, '<patched transcription_backend.py>', 'exec')
    except SyntaxError as exc:
        raise ValueError(f'patched transcription backend does not compile: {exc}') from exc
    check = ast.parse(result)
    patched_cls = next(n for n in check.body if isinstance(n, ast.ClassDef) and n.name == 'RealtimeTranscriber')
    if any(_is_flush_call(n) for n in ast.walk(patched_cls)):
        raise ValueError('flush call remains after patch')
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'mkdir'
           and _has_config_fallback_attr(n) for n in ast.walk(patched_cls)):
        raise ValueError('disk fallback mkdir remains after patch')
    return result

def patch_app(text):
    if 'class _RTSession' not in text or 'RT = _RTSession()' not in text or 'SAMPLE_RATE' not in text:
        raise ValueError('app RT anchors missing')
    start = 'if STATE.shutting_down:\n        return PlainTextResponse("shutting_down", status_code=409)'
    if start not in text:
        raise ValueError('start guard anchor missing')
    inject = '\n\n    if not RECORDING.start(SAMPLE_RATE):\n        return PlainTextResponse("recording_save_in_progress", status_code=409)'
    text = text.replace(start, start + inject, 1)
    feed = '    # Feed realtime transcriber with the current frame (non-blocking safety)'
    if text.count(feed) != 1:
        raise ValueError('feed anchor must be unique')
    text = text.replace(feed, '    RECORDING.append(frame)\n' + feed, 1)
    stop = '    await broadcast("status", "audio_stopped")'
    if text.count(stop) != 1:
        raise ValueError('stop broadcast anchor must be unique')
    text = text.replace(stop, '    RECORDING.stop()\n' + stop, 1)
    header = '# ---------- Control routes ----------'
    if header in text:
        at = text.index(header)
    else:
        needle = '@app.post("/start", response_class=PlainTextResponse)'
        if text.count(needle) != 1:
            raise ValueError('control insertion anchor missing')
        at = text.index(needle)
    text = text[:at] + APP_BLOCK + '\n' + text[at:]
    compile(text, '<patched app.py>', 'exec')
    return text


def patch_html(text):
    if not re.search(r'id\s*=\s*(["\'])btn-stop\1', text) or '</body>' not in text:
        raise ValueError('HTML anchors missing')
    return text.replace('</body>', HTML_BLOCK + '</body>', 1)


def main():
    ap = argparse.ArgumentParser(description='Apply bounded in-memory recording patch')
    ap.add_argument('repo', nargs='?', default='.', type=Path)
    args = ap.parse_args()
    root = args.repo.resolve()
    app = unique_named(root, 'app.py')
    trans = unique_named(root, 'transcription_backend.py')
    html = [p for p in files(root) if p.suffix.lower() in {'.html', '.htm'}
            and re.search(r'id\s*=\s*(["\'])btn-stop\1', p.read_text(encoding='utf-8', errors='strict'))
            and '</body>' in p.read_text(encoding='utf-8', errors='strict')]
    if not html:
        raise ValueError('no matching HTML files')
    targets = [app, trans, *html]
    # Read exact bytes first; strict decoding deliberately refuses lossy source transformations.
    original_bytes = {p: p.read_bytes() for p in targets}
    original = {p: b.decode('utf-8') for p, b in original_bytes.items()}
    marked = [MARK in original[p] for p in targets]
    if all(marked):
        print('Already patched; no files changed.')
        return
    if any(marked):
        raise ValueError('partial patch markers found; refusing to write')
    changed = {app: patch_app(original[app]), trans: patch_trans(original[trans])}
    changed.update({p: patch_html(original[p]) for p in html})
    for p, value in changed.items():
        if p.suffix == '.py':
            compile(value, str(p), 'exec')
        if MARK not in value:
            raise ValueError(f'marker absent in {p}')
    backups = []
    try:
        for p in targets:
            backup = p.with_name(p.name + '.recording_patch.bak')
            if backup.exists():
                raise ValueError(f'backup already exists: {backup}')
            shutil.copy2(p, backup)
            backups.append((p, backup))
        for p, value in changed.items():
            p.write_text(value, encoding='utf-8')
    except Exception:
        for p, backup in backups:
            shutil.copy2(backup, p)
        raise
    print('Patched ' + ', '.join(str(p.relative_to(root)) for p in targets)
          + '. Backups created; rerun is idempotent.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('Patch not applied: ' + str(exc), file=sys.stderr)
        sys.exit(1)
