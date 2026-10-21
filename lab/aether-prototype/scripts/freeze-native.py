"""Build one example with unchanged local sources and retain its input identity."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess

root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument("--example",default="h3_generate")
ap.add_argument("--bundle",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True)
args=ap.parse_args()
metadata=json.loads(subprocess.check_output(["cargo","metadata","--offline","--format-version","1"],cwd=root,text=True))
package=next(p for p in metadata["packages"] if p["name"]=="corsairs")
dependency=Path(package["manifest_path"]).parent
paths={}
for directory,prefix in [(root,"aether"),(dependency,"corsairs")]:
    bases=[directory/"src",directory/"examples",directory/"forge"] if prefix=="aether" else [directory/"src"]
    for base in bases:
        for path in base.rglob("*"):
            if path.is_file() and (path.suffix in (".rs",".toml",".py",".json") or path.name=="Cargo.toml"):
                paths[f"{prefix}/{path.relative_to(directory)}"]=path
    for filename in ["Cargo.toml","Cargo.lock"]:
        path=directory/filename
        if path.is_file():paths[f"{prefix}/{filename}"]=path
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={name:sha(path) for name,path in paths.items()}
subprocess.run(["cargo","build","--offline","--release","--example",args.example],cwd=root,check=True)
after={name:sha(path) for name,path in paths.items()}
if before!=after:raise RuntimeError("sources changed during build; do not launch this artifact")
output=args.output.resolve()
output.mkdir(parents=True,exist_ok=False)
binary=output/args.example
shutil.copy2(root/"target/release/examples"/args.example,binary)
for name,path in paths.items():
    target=output/"sources"/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target)
    if sha(target)!=before[name]:raise RuntimeError("source changed while retaining snapshot")
bundle=args.bundle.resolve()
shutil.copy2(bundle,output/"bundle.json")
manifest=dict(binary=str(binary),binary_sha256=sha(binary),bundle=str(bundle),bundle_sha256=sha(bundle),dependency=dict(name=package["name"],version=package["version"],source=package["source"]),
    source_sha256=before,rustc=subprocess.check_output(["rustc","--version"],text=True).strip(),
    cargo_features=[],scope="Unchanged-source build and retained snapshots; not an atomic OS image binding, host environment capture or framework release qualification.")
(output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(output/"manifest.json")
