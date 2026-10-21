import collections,datetime,hashlib,json,re,shutil
from pathlib import Path
root=Path.cwd();base=root/'.local/native-prepared-prompt-01';evidence=root/'docs/evidence'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((base/'report.json').read_text());pre=json.loads(Path('.local/native-prepared-prompt-01-provenance.json').read_text())
assert r['passed'] and r['stage']=='complete' and len(r['checks'])==65 and all(r['checks'].values()) and r['modelStreamsVerified']
for p,digest in pre['sourceSHA256'].items(): assert sha(root/p)==digest,p
log=Path('.local/native-prepared-prompt-01-tests.log').read_text()
assert 'error:' not in log and log.count(' skipped.')==3 and 'run with 58 tests' in log
assert re.search(r'nativeKreaDynamicParityAndStageCancellation\(\) passed after 510\.303 seconds',log)
counts=collections.Counter(e['event'] for e in r['events']);assert counts['nativeBegin']==counts['nativeEnd']==303
opened={e['instance']:e for e in r['events'] if e['event']=='modelOpened'}
closed={e['instance']:e for e in r['events'] if e['event']=='modelClosed'}
assert opened.keys()==closed.keys() and len(opened)==counts['modelOpened']==counts['modelClosed']==28
for instance in opened:
 end=max(e['sequence'] for e in r['events'] if e.get('instance')==instance and e['event']=='nativeEnd')
 assert opened[instance]['sequence']<end<closed[instance]['sequence']
bycase={};imagehashes={}
for case in r['cases']:
 cid=case['specification']['id'];ref=json.loads((base/f'{cid}-reference/record.json').read_text())['runs'][0];assert len(case['runs'])==4
 prior=json.loads((evidence/f'krea-function-reuse-reference-{cid}-2026-10-18.json').read_text())['runs'][0]
 for key in ['encodedTextSHA256','stepLatentsSHA256','pixelsSHA256','pixelShape']: assert ref[key]==prior[key],(cid,key)
 for run in case['runs']:
  assert run['conditioningSHA256']==ref['encodedTextSHA256'] and run['stepLatentsSHA256']==ref['stepLatentsSHA256'] and run['pixelsSHA256']==ref['pixelsSHA256'] and run['pixelShape']==ref['pixelShape']
  imagehashes[str((base/run['png']).relative_to(root))]=sha(base/run['png'])
 for repeat in [1,2]:
  events=[e for e in r['events'] if e['phase']==f'{cid}.preparedRepeat{repeat}']
  # Terminal unload follows repeat 2 under the same phase label. Bound the
  # generation window at its last completed model call before checking reuse.
  last_return=max(e['sequence'] for e in events if e['event']=='callReturned')
  events=[e for e in events if e['sequence']<=last_return]
  assert sum(e['event']=='nativeBegin' for e in events)==33
  assert all(e['detail'].startswith(('segment:', 'decode:')) for e in events if e['event']=='callSubmitted')
  if repeat==2:
   assert not any(e['event'] in ['modelOpened','modelClosed','functionLoadBegin','functionLoadEnd'] for e in events)
   assert sum(e['event']=='functionReused' for e in events)==33
 bycase[cid]={'width':case['specification']['width'],'height':case['specification']['height'],'steps':8,'seed':case['specification']['seed'],'conditioningSHA256':ref['encodedTextSHA256'],'stepLatentsSHA256':ref['stepLatentsSHA256'],'pixelsSHA256':ref['pixelsSHA256'],'completedGenerations':4,'preparedGenerations':2,'encoderReleasedBeforePreparedGeneration':True,'secondPreparedGenerationReusesAllModelsAndFunctions':True}
artifacts={}
def archive(src,name):
 dst=evidence/name;assert not dst.exists(),dst
 shutil.copyfile(src,dst);artifacts[str(dst.relative_to(root))]=sha(dst)
archive(base/'report.json','krea-prepared-prompt-native-2026-10-18.json')
for suffix,ext in [('tests','log'),('provenance','json'),('device','json'),('toolchain','txt')]:
 archive(root/f'.local/native-prepared-prompt-01-{suffix}.{ext}',f'krea-prepared-prompt-native-{suffix}-2026-10-18.{ext}')
for cid in bycase:
 archive(base/f'{cid}-reference/record.json',f'krea-prepared-prompt-reference-{cid}-2026-10-18.json')
 archive(base/f'{cid}-reference/reference.log',f'krea-prepared-prompt-reference-{cid}-2026-10-18.log')
summary={'schema':1,'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'scope':'Native qualification of prepared prompts at two dynamic shapes, independent eight-step parity, encoder/decoder cancellation recovery and session/model/function ownership. User confirmed GPU quiet; this is a correctness run, not a benchmark.','registeredTests':58,'passedCPUTests':54,'passedNativeTests':1,'skippedOtherNativeSuites':3,'nativeChecks':65,'nativeCalls':303,'nativeModelSessionsOpenedAndClosed':28,'nativeFunctionLoads':28,'nativeFunctionReuses':275,'libraryCompletedGenerations':8,'preparedGenerations':4,'independentReferenceGenerations':2,'nativeTestElapsedSeconds':510.303,'bundleFingerprint':r['bundleFingerprint'],'modelStreamsVerified':True,'currentSourcesMatchPreRunHashes':True,'independentReferencesMatchPreviousQualifiedHashes':True,'allNativeCallsReturnedBeforeModelClose':True,'referenceBinarySHA256':r['referenceBinarySHA256'],'sourceSHA256':pre['sourceSHA256'],'testExecutableSHA256':sha(root/'.build/arm64-apple-macosx/release/AetherKitPackageTests.xctest/Contents/MacOS/AetherKitPackageTests'),'deviceAfterRun':json.loads(Path('.local/native-prepared-prompt-01-device.json').read_text()),'checks':r['checks'],'cases':bycase,'artifactsSHA256':artifacts,'localPNGSHA256':imagehashes,'limits':['Two prompts and image shapes at seed 17017; native seed variation and broader shapes/devices are not qualified by this run.','Native cancellation/recovery covers encoded generation stages. Preparation-specific cancellation and cross-session prepared ownership are currently CPU-qualified.','Elapsed time includes stream verification, independent references, diagnostics and cancellation. No prepared-prompt performance benchmark was executed.','Native calls are observed across awaited framework execution; GPU queue draining is not independently instrumented.','Prepared prompt byte counts describe conditioning payload; model and process memory are separate.']}
(evidence/'krea-prepared-prompt-qualification-2026-10-18.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Verified 65 checks, 303 paired native calls, 28 model lifecycles, eight completed library images and unchanged source hashes.')
print('Second prepared repeat at each shape reused all 33 functions without any native model/function loads.')
