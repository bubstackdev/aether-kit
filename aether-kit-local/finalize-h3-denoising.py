import datetime,hashlib,json,collections
from pathlib import Path
root=Path.cwd();dest=root/'docs/evidence';prefix=root/'.local/h3-denoising-native-01'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
r=json.loads((prefix/'report.json').read_text());c=json.loads(Path(str(prefix)+'-controller.json').read_text())
p=json.loads((prefix/'progress.json').read_text());provenance=json.loads(Path(str(prefix)+'-provenance.json').read_text())
assert r['passed'] and r['stage']=='complete' and not r['captureComplete'] and r['nativeExecuted'] and r['readyForQualification']
assert c['passed'] and c['joined'] and not c['timedOut'] and c['exitCode']==0 and c['sourcePinsUnchanged']
assert r==json.loads(Path(str(prefix)+'-stdout.json').read_text()) and r['outputs']==p['outputs']
assert len(r['outputs'])==864 and len(p['comparisons'])==464
assert len({x['key'] for x in r['outputs']})==864
refs=json.loads((root/'.local/h3-denoising-cpu-references-01/references.json').read_text())
expected={x['key']:x['sha256'] for x in r['outputs']};assert expected==refs['expected']
comparisons={};groups=collections.defaultdict(list);full=sampled=0
for row in p['comparisons']:
 comparison=row.get('full') or row.get('sampled');assert comparison['passed'] and comparison['relativeL2']<=.02
 if 'full' in row:full+=1;assert comparison['actualSHA256']==expected[row['key']]
 else:sampled+=1
 comparisons[row['key']]=comparison;groups[row['key'].split('.')[0]].append(comparison['relativeL2'])
assert full==448 and sampled==16
for name,digest in provenance['filesSHA256'].items():assert sha(name)==digest,name
capture_provenance=json.loads((dest/'h3-denoising-capture-provenance-2026-10-18.json').read_text())['filesSHA256']
assert all(provenance['filesSHA256'][name]==digest for name,digest in capture_provenance.items())
assert len(provenance['filesSHA256'])==len(capture_provenance)+1
peak=r['sampledPeak'];assert peak['processLifetimePeakFootprintBytes']<=20<<30 and peak['metalAllocatedBytes']<=18<<30
assert not Path(str(prefix)+'-stderr.log').read_text()
artifacts={'native':prefix/'report.json','native-progress':prefix/'progress.json','native-controller':Path(str(prefix)+'-controller.json'),'native-provenance':Path(str(prefix)+'-provenance.json'),'native-stderr':Path(str(prefix)+'-stderr.log'),
 'reference-prepared':root/'.local/h3-denoising-reference-prepared-01/report.json','reference-preparation-controller':root/'.local/h3-denoising-reference-prepared-01-controller.json','reference-preparation-provenance':root/'.local/h3-denoising-reference-prepared-01-provenance.json','final-compiler-interop':root/'.local/h3-denoising-tests-06.log'}
for label,source in artifacts.items():
 suffix='.log' if label in ['native-stderr','final-compiler-interop'] else '.json';target=dest/f'h3-denoising-{label}-2026-10-18{suffix}'
 with target.open('xb') as f:f.write(source.read_bytes())
summary=dict(schema=1,recordedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,
 scope=r['scope'],providedConditioning=True,geometry=dict(width=960,height=544,frames=124,packedRows=19333),
 cpuTests=dict(passed=112,registered=116,nativeTestsDisabled=4),
 nativeQualification=dict(submissions=824,steps=8,backboneBlocksPerStep=50,modulationCalls=400,finiteOutputIdentities=864,fullTensorCPUComparisons=448,sampledBodyComparisons=16,relativeL2Limit=.02,
 allCapturedHashesReproduced=True,maximumRelativeL2=max(v['relativeL2'] for v in comparisons.values()),
 byStage={k:dict(count=len(v),maximumRelativeL2=max(v)) for k,v in groups.items()},
 bodyRelativeL2ByStep={str(s):{str(b):comparisons[f'body.{s}.{b}.hidden']['relativeL2'] for b in [0,49]} for s in range(8)},
 finalLatentSHA256={name:expected[f'latents.7.-1.{name}'] for name in ['video','audio']}),
 sampledPeak=peak,memorySamples=r['memorySamples'],resources=r['resources'],controller=c,
 captureNativeSubmissions=824,failedMemoryControlNativeSubmissions=1,totalNativeSubmissionsIncludingControl=1649,
 memoryReaderFixValidated=True,sharedCaptureAndQualificationPinsIdentical=True,
 limitations=[
 'Provided conditioning/latents at one fixed T2VA geometry only. This does not implement conditioner presentation/unnormalized taps, keyframes, VAEs, stitching, encoded media or quality acceptance.',
 'CPU comparisons use complete captured native input contexts. Only 64 output rows at blocks 0 and 49 are compared at each step; this is not all-block or independent full CPU trajectory parity.',
 'Capture/replay reproduce hashes across separate owned processes. Native whole-trajectory cancellation/recovery, retained full-run repetition and long-run/device-wide retention remain separate gates; CPU session cancellation/backpressure tests and earlier per-module native gates do not establish those.',
 'Boundary snapshots use CPU-owned copies between stages; this is not a zero-copy pipeline or a benchmark.',
 'Memory counters overlap, 20 ms sampling can miss Metal peaks, and observed memory bounds do not establish general deployment guarantees.',
 'Eight SDK stream Sendable diagnostics remain across four confined experimental adapters; no unchecked Sendable conformance was added.',
 'Elapsed controller times include verification, CPU comparison, file IO and settling on a shared machine and are not performance measurements.'
 ],nextGate='Qualify native whole-trajectory cancellation/recovery and retained repetition, then integrate real conditioner presentation/taps and both decoders before media acceptance.')
summary['artifactSHA256']={p.name:sha(p) for p in sorted(dest.glob('h3-denoising-*-2026-10-18.*'))}
with (dest/'h3-denoising-qualification-2026-10-18.json').open('x') as f:f.write(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(passed=True,maxRelativeL2=summary['nativeQualification']['maximumRelativeL2'],byStage=summary['nativeQualification']['byStage'],peakFootprintGiB=peak['processLifetimePeakFootprintBytes']/(1<<30),sampledMetalGiB=peak['metalAllocatedBytes']/(1<<30),settledFootprintGiB=r['resources']['afterUnload5s']['physicalFootprintBytes']/(1<<30),bodyRelativeL2=summary['nativeQualification']['bodyRelativeL2ByStep'],artifacts=len(summary['artifactSHA256'])),indent=2))
