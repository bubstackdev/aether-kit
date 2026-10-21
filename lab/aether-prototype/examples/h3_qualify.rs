//! Read-only local input gate by default; native qualification is explicit.
use aether::h3::H3Lab;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=4).contains(&args.len())
        || args.get(3).is_some_and(|a| {
            !["--native", "--schedule", "--backbone", "--backbone-inputs"].contains(&a.as_str())
        })
    {
        return Err("usage: h3_qualify H3_LAB REPORT.json [ROWS [--native|--schedule|--backbone|--backbone-inputs]]".into());
    }
    let rows = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(128);
    let mode = args.get(3).map(String::as_str).unwrap_or("");
    let native = matches!(mode, "--native" | "--schedule" | "--backbone");
    let schedule = mode == "--schedule";
    let backbone = matches!(mode, "--backbone" | "--backbone-inputs");
    if schedule && rows != 128 {
        return Err("the retained H3 schedule gate is block 0 at 128 rows".into());
    }
    if backbone && rows != 19350 {
        return Err("the retained H3 synthetic backbone gate requires 19350 rows".into());
    }
    let lab = H3Lab::open(&args[0])?;
    let inputs = if backbone {
        lab.inspect_backbone()?
    } else {
        lab.inspect_inputs(rows)?
    };
    let mut report = serde_json::json!({"schema":1,"native_requested":native,"inputs":inputs,"generation_available":false});
    let mut failed = false;
    if native {
        let result = corsairs::Runtime::new()
            .map_err(aether::Error::from)
            .and_then(|runtime| {
                futures_executor::block_on(async {
                    if backbone {
                        Ok(serde_json::to_value(lab.qualify_backbone(&runtime).await?)?)
                    } else if schedule {
                        Ok(serde_json::to_value(lab.qualify_schedule(&runtime).await?)?)
                    } else {
                        Ok(serde_json::to_value(
                            lab.qualify_block(&runtime, rows).await?,
                        )?)
                    }
                })
            });
        match result {
            Ok(result) => {
                failed = result["passed"] != true;
                report["native"] = result;
            }
            Err(error) => {
                failed = true;
                report["native"] = serde_json::json!({"passed":false,"error":error.to_string()});
            }
        }
    }
    let path = Path::new(&args[1]);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", path.display());
    if failed {
        return Err("H3 native qualification failed; see report".into());
    }
    Ok(())
}
