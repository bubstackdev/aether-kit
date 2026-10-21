import datetime,hashlib,json
from pathlib import Path
root=Path.cwd();stem='h3-denoising-lifecycle';date='2026-10-18';native=root/'.local'/f'{stem}-native-01';prepared=root/'.local'/f'{stem}-prepared-01';dest=root/'docs/evidence';artifacts={}
def read(p):return json.loads(p.read_text())
def archive(p,label):
 target=dest/f'{stem}-interrupted-{label}-{date}{p.suffix}'
 with target.open('xb') as f:f.write(p.read_bytes())
 artifacts[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
controller=read(Path(str(native)+'-controller.json'));report=read(native/'report.json')
assert controller['joined'] and controller['exitCode']==-15 and not controller['passed'] and not controller['timedOut'] and controller['sourcePinsUnchanged']
roles=['initial','retainedCancellation','retainedRecovery','closingCancellation','reopenedRecovery'];progress={}
refs=read(root/'.local/h3-denoising-cpu-references-01/references.json')['expected']
for role in roles:
 p=native/role/'progress.json';d=read(p)
 assert all(x['sha256']==refs[x['key']] for x in d['outputs'])
 assert all((x.get('full') or x['sampled'])['passed'] for x in d['comparisons'])
 progress[role]=dict(outputs=len(d['outputs']),CPUComparisons=len(d['comparisons']),lastOutput=d['outputs'][-1]['key'],completed=role in report['lifecycle']['runs'])
 archive(p,role+'-progress')
archive(native/'report.json','native-report')
archive(prepared/'report.json','prepared-report')
for folder,label in [(native,'native'),(prepared,'prepared')]:
 for suffix in ['controller.json','provenance.json','stderr.log']:archive(Path(str(folder)+'-'+suffix),label+'-'+suffix.rsplit('.',1)[0])
archive(root/'.local/h3-denoising-lifecycle-tests-02.log','tests')
archive(root/'Scripts/qualify-h3-denoising.py','controller-script')
summary=dict(schema=1,recordedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='interruptedAtUserRequest',passed=False,reason='User requested cancellation, a handoff document and spin down. No restart authorized.',controllerJoined=True,exitCode=controller['exitCode'],timedOut=False,sourcePinsUnchanged=True,elapsedSeconds=controller['elapsedSeconds'],completedRoles=report['lifecycle']['runs'],progress=progress,lastPersistedSampledPeak=report['sampledPeak'],resources=report['resources'],remaining=['Complete reopened eight-step recovery','Second-session unload/settlement and 512 MiB growth gate','Final complete lifecycle report and qualification archive'],artifactsSHA256=artifacts)
p=dest/f'{stem}-interruption-{date}.json'
with p.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in summary.items() if k!='artifactsSHA256'},indent=2))
