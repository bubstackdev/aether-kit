import hashlib,json
from pathlib import Path
root=Path.cwd();local=root/'.local';evidence=root/'docs/evidence';stem='h3-denoising-lifecycle';date='2026-10-18'
native=local/(stem+'-native-01');prepared=local/(stem+'-prepared-01')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
report=read(native/'report.json');controller=read(Path(str(native)+'-controller.json'));prep=read(Path(str(prepared)+'-controller.json'))
assert report['passed'] and report['stage']=='complete'
for c in [controller,prep]:assert c['passed'] and c['joined'] and c['exitCode']==0 and not c['timedOut'] and c['sourcePinsUnchanged']
assert not prep['gpuWorkRequested'] and controller['gpuWorkRequested']
roles=['initial','retainedCancellation','retainedRecovery','closingCancellation','reopenedRecovery']
assert report['lifecycle']['runs']==roles
refs=read(local/'h3-denoising-cpu-references-01/references.json');submitted=0;outputs=0;comparisons=0;full=0;sampled=0;maxL2=0;latent_exact=0
for role in roles:
 p=read(native/role/'progress.json');cancel='Cancellation' in role
 assert len(p['outputs'])==(435 if cancel else 864)
 assert len(p['comparisons'])==(410 if cancel else 464)
 assert all(o['sha256']==refs['expected'][o['key']] for o in p['outputs'])
 for c in p['comparisons']:
  v=c.get('full') or c['sampled'];assert v['passed'] and v['relativeL2']<=.02
  full+=int('full' in c);sampled+=int('sampled' in c);maxL2=max(maxL2,v['relativeL2'])
  if c['key'].startswith('latents.'):assert v['byteIdentical'];latent_exact+=1
 events=report['lifecycle']['traces'][role];count=sum(e.endswith(':nativeSubmitted') for e in events)
 assert count==(434 if cancel else 824)
 assert count==sum(e.endswith(':nativeCompleted') for e in events)
 submitted+=count;outputs+=len(p['outputs']);comparisons+=len(p['comparisons'])
assert (submitted,outputs,comparisons,full,sampled,latent_exact)==(3340,3462,2212,2162,50,48)
assert report['sampledPeak']['physicalFootprintBytes']<=20<<30 and report['sampledPeak']['metalAllocatedBytes']<=18<<30
baseline=report['resources']['firstSession.afterUnload5s'];settled=report['resources']['reopenedSession.afterUnload5s']
growth={k:settled[k]-baseline[k] for k in ['physicalFootprintBytes','residentBytes','metalAllocatedBytes']}
assert max(growth.values())<=512<<20
pins=read(Path(str(native)+'-provenance.json'))['filesSHA256'];prepared_pins=read(Path(str(prepared)+'-provenance.json'))['filesSHA256']
assert pins==prepared_pins and all(sha(Path(p))==h for p,h in pins.items())
oldpins=read(local/'h3-denoising-native-01-provenance.json')['filesSHA256']
changed=sorted(p for p in oldpins.keys() & pins.keys() if oldpins[p]!=pins[p]);added=sorted(pins.keys()-oldpins.keys());assert not oldpins.keys()-pins.keys()
assert all('/AetherH3CoreAI/' not in p and '/AetherH3/' not in p for p in changed+added)
artifacts={}
def archive(source,label):
 destination=evidence/f'{stem}-{label}-{date}{source.suffix}'
 with destination.open('xb') as f:f.write(source.read_bytes())
 artifacts[destination.name]=sha(destination)
archive(native/'report.json','native')
for folder,label in [(native,'native'),(prepared,'preparation')]:
 if label=='preparation':archive(folder/'report.json','prepared')
 for suffix in ['controller.json','provenance.json','stderr.log']:archive(Path(str(folder)+'-'+suffix),label+'-'+suffix.rsplit('.',1)[0])
for role in roles:archive(native/role/'progress.json',role+'-progress')
archive(local/(stem+'-tests-02.log'),'tests')
archive(root/'Scripts/qualify-h3-denoising.py','controller-script')
summary=dict(schema=1,passed=True,scope=report['scope'],nativeSubmissions=submitted,completeRuns=3,cancelledRuns=2,outputHashesReproduced=outputs,CPUComparisons=comparisons,fullTensorComparisons=full,sampledBodyComparisons=sampled,byteIdenticalLatentUpdates=latent_exact,maximumRelativeL2=maxL2,relativeL2Limit=.02,resources=report['resources'],sampledPeak=report['sampledPeak'],settledGrowthBytes=growth,sourcePins=len(pins),sourcePinsUnchanged=True,changedPinsFromPreviousQualification=changed,addedPins=added,CPUChecks=dict(passing=113,registered=117,nativeDisabled=4),controllerElapsedSeconds=controller['elapsedSeconds'],budgetSeconds=controller['budgetSeconds'],artifactsSHA256=artifacts)
destination=evidence/f'{stem}-qualification-{date}.json'
with destination.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in summary.items() if k!='artifactsSHA256'},indent=2))
