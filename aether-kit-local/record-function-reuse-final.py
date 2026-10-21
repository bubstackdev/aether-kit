import collections, hashlib, json, re, shutil, subprocess
from pathlib import Path

root = Path('docs/evidence')
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
artifacts = {}
def copy(src, name):
    dst = root / name
    shutil.copyfile(src, dst)
    artifacts[name] = digest(dst)
    return name

specs = {
    'full': ('native-dynamic-function-reuse-20261018', 39),
    'retained': ('native-recovery-retain-final-384-20261018', 10),
    'reopened': ('native-recovery-reopen-final-128-20261018', 11),
    'lifecycle': ('native-session-function-reuse-20261018', 23),
}
reports, runs = {}, {}
for key, (directory, checks) in specs.items():
    p = Path('.local') / directory
    r = read(p / 'report.json')
    assert r['passed'] and len(r['checks']) == checks and all(r['checks'].values()), key
    events = r['events']
    opened = sorted(x['instance'] for x in events if x['event'] == 'modelOpened')
    closed = sorted(x['instance'] for x in events if x['event'] == 'modelClosed')
    assert opened == closed and len(set(opened)) == len(opened), key
    counts = collections.Counter(x['event'] for x in events)
    begins = {x['call']: x for x in events if x['event'] == 'nativeBegin'}
    ends = [x for x in events if x['event'] == 'nativeEnd']
    assert len(begins) == len(ends), key
    for end in ends:
        begin = begins[end['call']]
        assert begin['sequence'] < end['sequence'] and begin['instance'] == end['instance']
    assert counts['functionLoadEnd'] == len(opened)
    assert counts['functionLoadEnd'] + counts['functionReused'] == len(begins)
    if (p / 'events.jsonl').exists():
        journal = [json.loads(line) for line in (p / 'events.jsonl').read_text().splitlines()]
        assert journal == events, key
    report_name = copy(p / 'report.json', f'krea-function-reuse-{key}-2026-10-18.json')
    log_name = copy(Path(str(p) + '.log'), f'krea-function-reuse-{key}-tests-2026-10-18.log')
    log = (root / log_name).read_text()
    passed = len(re.findall(r'^✔ Test (?!run with ).+ passed after ', log, re.M))
    skipped = len(re.findall(r'^➜ Test .+ skipped\.', log, re.M))
    assert passed == 36 and skipped == 3, (key, passed, skipped)
    reports[key] = r
    runs[key] = {'passed': True, 'checks_passed': checks, 'report': report_name, 'log': log_name,
        'registered_tests': 39, 'passed_tests': passed, 'cpu_tests_passed': 35, 'other_native_tests_skipped': skipped,
        'native_calls_completed': len(ends), 'model_sessions_opened_and_closed': len(opened),
        'function_loads': counts['functionLoadEnd'], 'function_reuses': counts['functionReused'],
        'journal_matches_report': True if (p / 'events.jsonl').exists() else None}

for case in ('dynamic512', 'dynamic384'):
    copy(Path('.local/native-dynamic-function-reuse-20261018') / (case + '-reference') / 'record.json',
         f'krea-function-reuse-reference-{case}-2026-10-18.json')
full = reports['full']
for path, expected in full['referenceSourceSHA256'].items(): assert digest(path) == expected
assert digest('/Users/zakkeown/Code/atelier/lab/krea-lab/bin/round2-final') == full['referenceBinarySHA256']
old = read('.local/native-session-balanced-20261017/report.json')
old_images = {x['name']: (x['latentSHA256'], x['pixelsSHA256']) for x in old['generations']}
new_images = {x['name']: (x['latentSHA256'], x['pixelsSHA256']) for x in reports['lifecycle']['generations']}
assert old_images == new_images and len(new_images) == 7
control = read('.local/native-recovery-retain-384-20261018/report.json')
experiment = read('.local/native-recovery-functions-384-20261018/report.json')
assert control['baselinePixelsSHA256'] == experiment['baselinePixelsSHA256'] == reports['retained']['baselinePixelsSHA256']
for kind in ('retain-128-observed', 'reopen-128'):
    r = read(f'.local/native-recovery-{kind}-20261018/report.json')
    assert r['passed'] and r['baselinePixelsSHA256'] == reports['reopened']['baselinePixelsSHA256']
    copy(f'.local/native-recovery-{kind}-20261018/report.json', f'krea-function-reuse-control-{kind}-2026-10-18.json')
copy('.local/native-recovery-retain-128-20261018/report.json', 'krea-function-reuse-observation-miss-2026-10-18.json')
smokes = {}
for policy in ('cpuOnly', 'automatic', 'preferGPU'):
    p = Path(f'.local/native-function-reuse-smoke-{policy}-20261018.json')
    r = read(p); assert r['passed'] and r['output'] == [-3, 1, 3, 7]
    smokes[policy] = copy(p, f'krea-function-reuse-smoke-{policy}-2026-10-18.json')
for p in root.glob('krea-function-reuse-*2026-10-18*'):
    if p.name != 'krea-function-reuse-qualification-2026-10-18.json': artifacts[p.name] = digest(p)
sources = sorted([*Path('Sources').rglob('*.swift'), *Path('Tests').rglob('*.swift'),
    Path('Package.swift'), Path('Package.resolved'), Path('Scripts/check.sh')])
q = {'schema': 1, 'date': '2026-10-18', 'passed': True, 'benchmark': False,
    'status': 'Lazy per-session function reuse passes independent full dynamic parity, cancellation/recovery and lifecycle regressions',
    'implementation': 'Resolve each requested stateless native function lazily once per model session; hold it until the backend/model is released at unload. Internal call-lifetime control and optional boundary observers remain available to tracked native tests.',
    'claims': {'independent_eight_step_parity_both_dynamic_shapes': True,
        'native_encoder_cancellation_and_full_reencoding_recovery': True,
        'native_decoder_cancellation_and_full_eight_step_recovery': True,
        'function_loads_once_per_model_session': True,
        'reopened_pipeline_recovery': True, 'shape_switch_and_active_unload': True,
        'seven_lifecycle_images_match_prechange_hashes': True, 'independent_sources_and_executable_unchanged': True},
    'runs': runs, 'affine_smoke': smokes,
    'independent_cases': [{'specification': c['specification'], 'expected': c['expected'], 'runs': c['runs']} for c in full['cases']],
    'diagnosis': {'control': 'Original per-call loading completed the baseline, then slowed during repeated denoising before cancellation was requested. The bounded control was operator-terminated after 43 completed native calls; its last native boundary was denoiser2 in the repeat.',
        'control_sample': 'krea-function-reuse-control-2026-10-18.sample.txt',
        'control_journal': 'krea-function-reuse-control-2026-10-18.jsonl',
        'memory_snapshot_peak_control': '35.3G (vmmap units)', 'memory_snapshot_peak_retained_function_experiment': '3.1G (vmmap units)',
        'experimental_arm': 'Exploratory eager function retention, before the final lazy implementation; completed 99 calls with identical baseline/recovery pixels. Final lazy implementation is qualified by the runs above.',
        'observation_miss': 'The initial 128x128 probe missed a roughly 10 ms native decode at the existing observation interval. A 1 ms observation interval then observed real in-flight cancellation, without gating native work.',
        'inference_limit': 'The comparison supports correcting native function lifetime at the application boundary; the framework allocator/driver mechanism is not isolated. Memory snapshots are diagnostic, with uncontrolled host/cache conditions.'},
    'source_sha256': {str(p): digest(p) for p in sources},
    'test_binary_sha256': digest('.build/arm64-apple-macosx/release/AetherKitPackageTests.xctest/Contents/MacOS/AetherKitPackageTests'),
    'toolchain': subprocess.check_output(['swift', '--version'], text=True).strip(),
    'bundle_fingerprint': full['bundleFingerprint'], 'model_streams_verified': full['modelStreamsVerified'],
    'cache_state': 'Uncontrolled OS cache; assets read from existing Hugging Face cache paths',
    'host_load': 'Shared host; quiet conditions not established; no latency or throughput claim',
    'limits': full['limits'] + ['Physical GPU queue draining and broader resident-encoder/device/failure coverage are not asserted'],
    'records': artifacts}
p = root / 'krea-function-reuse-qualification-2026-10-18.json'
p.write_text(json.dumps(q, indent=2) + '\n')
for name, expected in artifacts.items(): assert digest(root / name) == expected
print(json.dumps({'manifest': str(p), 'runs': runs, 'all_artifact_digests_verified': True}, indent=2))
