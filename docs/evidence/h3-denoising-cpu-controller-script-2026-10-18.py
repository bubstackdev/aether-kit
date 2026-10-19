import datetime, hashlib, json, os, pathlib, signal, subprocess, time
root=pathlib.Path.cwd();prefix=root/'.local/h3-denoising-cpu-references-01'
capture=root/'.local/h3-denoising-capture-02'
assert not prefix.exists()
assert not pathlib.Path(str(prefix)+'-controller.json').exists()
controller=json.loads(pathlib.Path(str(capture)+'-controller.json').read_text())
assert controller['passed'] and controller['joined'] and not controller['timedOut'] and controller['sourcePinsUnchanged']
r=json.loads((capture/'report.json').read_text());assert r['captureComplete'] and not r['passed'] and len(r['outputs'])==864
python='/Users/zakkeown/Code/aether-engine/forge/.venv/bin/python'
command=[python,'-B',str(root/'Scripts/prepare-h3-denoising-oracle.py'),'--lab','/Users/zakkeown/Code/atelier/lab/minimax-h3-lab',
 '--prototype','/Users/zakkeown/Code/bubstackdev/aether/lab/aether-prototype','--bundle','/private/tmp/aether-h3-qkv-ready/render/bundle.json',
 '--phases','/private/tmp/aether-h3-qkv-ready/phase/bundle.json','--cpu-record','/private/tmp/aether-h3-qkv-phase-oracles/cpu-checks.json',
 '--capture',str(capture),'--output',str(prefix)]
budget=1800
state=dict(schema=1,startedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,budgetSeconds=budget,gpuWork=False,passed=False,joined=False,timedOut=False)
path=pathlib.Path(str(prefix)+'-controller.json')
def save():path.write_text(json.dumps(state,indent=2)+'\n')
start=time.monotonic()
with open(str(prefix)+'-stdout.log','xb') as stdout,open(str(prefix)+'-stderr.log','xb') as stderr:
 proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True);state['pid']=proc.pid;save();print(json.dumps(state),flush=True)
 try:proc.wait(timeout=budget-10)
 except subprocess.TimeoutExpired:state['timedOut']=True
 finally:
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=8)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
  proc.wait();state.update(joined=True,exitCode=proc.returncode,elapsedSeconds=time.monotonic()-start);save()
try:
 r=json.loads((prefix/'cpu-checks.json').read_text());state['passed']=proc.returncode==0 and not state['timedOut'] and r['passed'] and not r['gpu_work'] and r['references']==464
except (OSError,ValueError) as error:state['reportError']=str(error)
save();print(json.dumps(state,indent=2),flush=True)
raise SystemExit(0 if state['passed'] else 1)
