import datetime, hashlib, json, os, pathlib, signal, subprocess, time
root = pathlib.Path.cwd()
prefix = root / '.local/h3-lifecycle-native-02'
controller_path = pathlib.Path(str(prefix) + '-controller.json')
provenance_path = pathlib.Path(str(prefix) + '-provenance.json')
exe = root / '.build/arm64-apple-macosx/release/aether-probe'
bundle = pathlib.Path('/private/tmp/aether-h3-qkv-ready/render/bundle.json')
phases = pathlib.Path('/private/tmp/aether-h3-qkv-ready/phase/bundle.json')
def sha(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()
prepared = json.loads((root / '.local/h3-lifecycle-prepared-03/report.json').read_text())
assert prepared['passed'] and not prepared['nativeExecuted'] and prepared['stage'] == 'ready'
assert sha(bundle) == prepared['bundleSHA256']
assert sha(phases) == prepared['phaseManifestSHA256']
assert not prefix.exists()
assert not controller_path.exists() and not provenance_path.exists()
paths = [root / 'Package.swift', exe, bundle, phases, root / '.local/h3-lifecycle-stream-reuse-tests-final.log', root / '.local/h3-lifecycle-prepared-03/report.json']
for folder in ['Sources/AetherH3', 'Sources/AetherH3CoreAI', 'Sources/AetherProbe', 'Tests/AetherH3CoreAITests', 'Tests/AetherProbeTests']:
    paths.extend(sorted((root / folder).rglob('*.swift')))
pins = {str(p): sha(p) for p in paths}
provenance_path.write_text(json.dumps({'schema': 1, 'filesSHA256': pins}, indent=2) + '\n')
command = [str(exe), 'qualify-h3-lifecycle', str(bundle), str(phases), str(prefix), '--run']
controller = {'startedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': command, 'budgetSeconds': 300, 'passed': False, 'joined': False, 'timedOut': False}
start = time.monotonic()
with open(str(prefix) + '-stdout.json', 'xb') as stdout, open(str(prefix) + '-stderr.log', 'xb') as stderr:
    proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
    controller['pid'] = proc.pid
    controller_path.write_text(json.dumps(controller, indent=2) + '\n')
    print(json.dumps({'pid': proc.pid, 'output': str(prefix), 'budgetSeconds': 300}), flush=True)
    while proc.poll() is None and time.monotonic() - start < 290:
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    if proc.poll() is None:
        controller['timedOut'] = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
controller.update(joined=True, exitCode=proc.returncode, elapsedSeconds=time.monotonic()-start)
controller['sourcePinsUnchanged'] = all(sha(pathlib.Path(p)) == digest for p, digest in pins.items())
report_path = prefix / 'report.json'
report = json.loads(report_path.read_text()) if report_path.exists() else {}
controller['passed'] = proc.returncode == 0 and not controller['timedOut'] and report.get('passed') is True and controller['sourcePinsUnchanged']
controller_path.write_text(json.dumps(controller, indent=2) + '\n')
print(json.dumps(controller, indent=2), flush=True)
raise SystemExit(0 if controller['passed'] else 1)
