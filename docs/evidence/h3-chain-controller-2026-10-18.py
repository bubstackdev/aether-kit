import datetime, hashlib, json, os, pathlib, signal, subprocess, sys, time
root=pathlib.Path.cwd()
mode,chain_name,prepared_name,output_name=sys.argv[1:]
assert mode in ['capture','run']
chain=(root/chain_name).resolve(); prepared=(root/prepared_name).resolve()
prefix=(root/output_name).resolve(); exe=root/'.build/arm64-apple-macosx/release/aether-probe'
bundle=pathlib.Path('/private/tmp/aether-h3-qkv-ready/render/bundle.json')
phases=pathlib.Path('/private/tmp/aether-h3-qkv-ready/phase/bundle.json')
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=json.loads(prepared.read_text())
assert p['passed'] and p['stage']=='ready' and p['selectedFilesVerified'] and not p['nativeExecuted']
assert p['chainManifestSHA256']==sha(chain) and p['bundleSHA256']==sha(bundle) and p['phaseManifestSHA256']==sha(phases)
if mode=='run':assert p['readyForQualification']
assert not prefix.exists()
controller_path=pathlib.Path(str(prefix)+'-controller.json')
provenance_path=pathlib.Path(str(prefix)+'-provenance.json')
assert not controller_path.exists() and not provenance_path.exists()
paths=[exe,bundle,phases,chain,prepared,root/'Package.swift',root/'Scripts/prepare-h3-chain-oracle.py',root/'.local/h3-chain-tests-final.log']
for folder in ['Sources/AetherH3','Sources/AetherH3CoreAI','Sources/AetherProbe','Tests/AetherH3Tests','Tests/AetherH3CoreAITests','Tests/AetherProbeTests']:
    paths+=sorted((root/folder).rglob('*.swift'))
pins={str(p):sha(p) for p in paths}
provenance_path.write_text(json.dumps({'schema':1,'filesSHA256':pins},indent=2)+'\n')
command=[str(exe),'qualify-h3-chain',str(bundle),str(phases),str(chain),str(prefix),'--'+mode]
controller={'startedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'budgetSeconds':300,'passed':False,'joined':False,'timedOut':False}
start=time.monotonic()
with open(str(prefix)+'-stdout.json','xb') as stdout,open(str(prefix)+'-stderr.log','xb') as stderr:
    proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
    controller['pid']=proc.pid
    controller_path.write_text(json.dumps(controller,indent=2)+'\n')
    print(json.dumps({'pid':proc.pid,'mode':mode,'budgetSeconds':300}),flush=True)
    while proc.poll() is None and time.monotonic()-start<290:
        try:proc.wait(timeout=1)
        except subprocess.TimeoutExpired:pass
    if proc.poll() is None:
        controller['timedOut']=True;os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
    proc.wait()
controller.update(joined=True,exitCode=proc.returncode,elapsedSeconds=time.monotonic()-start)
controller['sourcePinsUnchanged']=all(sha(pathlib.Path(p))==digest for p,digest in pins.items())
r=json.loads((prefix/'report.json').read_text()) if (prefix/'report.json').exists() else {}
accepted=(r.get('captureComplete') is True and r.get('passed') is False) if mode=='capture' else (r.get('passed') is True and r.get('captureComplete') is False)
controller['passed']=proc.returncode==0 and not controller['timedOut'] and accepted and controller['sourcePinsUnchanged']
controller_path.write_text(json.dumps(controller,indent=2)+'\n');print(json.dumps(controller,indent=2),flush=True)
raise SystemExit(0 if controller['passed'] else 1)
