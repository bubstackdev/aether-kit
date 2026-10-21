import json
from pathlib import Path
root=Path('.local/h3-denoising-lifecycle-native-01')
for path in [Path(str(root)+'-controller.json'),root/'report.json']:
    if path.exists():
        data=json.loads(path.read_text());print(path.name,{k:data.get(k) for k in ['pid','stage','passed','joined','error','elapsedSeconds','sampledPeak'] if k in data})
for path in sorted(root.glob('*/progress.json')):
    data=json.loads(path.read_text());outputs=data['outputs'];comp=data['comparisons'];bad=[x['key'] for x in comp if not (x.get('full') or x.get('sampled'))['passed']]
    print(path.parent.name,'outputs',len(outputs),'comparisons',len(comp),'last',outputs[-1]['key'] if outputs else None,'failed',bad)
