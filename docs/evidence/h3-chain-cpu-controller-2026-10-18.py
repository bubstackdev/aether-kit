import datetime,json,os,pathlib,signal,subprocess,time
root=pathlib.Path.cwd(); prefix=root/'.local/h3-chain-cpu-references-01'
assert not prefix.exists()
command=['/Users/zakkeown/Code/aether-engine/forge/.venv/bin/python','-B','Scripts/prepare-h3-chain-oracle.py','--lab','/Users/zakkeown/Code/atelier/lab/minimax-h3-lab','--prototype','/Users/zakkeown/Code/bubstackdev/aether/lab/aether-prototype','--bundle','/private/tmp/aether-h3-qkv-ready/render/bundle.json','--phases','/private/tmp/aether-h3-qkv-ready/phase/bundle.json','--output',str(prefix),'--references',str(root/'.local/h3-chain-capture-01')]
report={'startedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'budgetSeconds':600,'gpuWork':False,'passed':False,'joined':False,'timedOut':False}
path=pathlib.Path(str(prefix)+'-controller.json'); assert not path.exists()
start=time.monotonic()
with open(str(prefix)+'-stdout.log','xb') as stdout,open(str(prefix)+'-stderr.log','xb') as stderr:
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True,env=env)
    report['pid']=proc.pid;path.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'pid':proc.pid,'gpuWork':False,'budgetSeconds':600}),flush=True)
    while proc.poll() is None and time.monotonic()-start<590:
        try:proc.wait(timeout=1)
        except subprocess.TimeoutExpired:pass
    if proc.poll() is None:
        report['timedOut']=True;os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
    proc.wait()
report.update(joined=True,exitCode=proc.returncode,elapsedSeconds=time.monotonic()-start)
cpu=json.loads((prefix/'cpu-checks.json').read_text()) if (prefix/'cpu-checks.json').exists() else {}
report['passed']=proc.returncode==0 and not report['timedOut'] and cpu.get('passed') is True
path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
raise SystemExit(0 if report['passed'] else 1)
