use aether::{
    ImagePipeline, ImageRequest, ImageSize,
    krea::{KreaAssets, KreaOptions, KreaPipeline, ShapeSpecialization},
};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=8).contains(&args.len()) {
        return Err(
            "usage: render BUNDLE OUTPUT.png PROMPT [WIDTH HEIGHT SEED [SHAPE.json|- [REPEATS]]]"
                .into(),
        );
    }
    let width = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(768);
    let height = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(1344);
    let seed = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(42);
    let repeats: usize = args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(1);
    if repeats == 0 {
        return Err("REPEATS must be positive".into());
    }
    let mut assets = KreaAssets::open(&args[0])?;
    if let Some(shape) = args.get(6).filter(|s| *s != "-") {
        eprintln!("Verifying original and specialized source streams...");
        assets.add_specialization(ShapeSpecialization::from_file(shape)?)?;
    }
    let mut pipeline = KreaPipeline::new(assets, KreaOptions::default())?;
    let request = ImageRequest::new(&args[2], ImageSize::new(width, height), seed);
    let prompt = pipeline.prepare_prompt(&request.prompt)?;
    eprintln!(
        "{} tokens, {} conditioning rows, truncated={}",
        prompt.token_ids().len(),
        prompt.text_rows(),
        prompt.truncated()
    );
    futures_executor::block_on(async {
        for index in 0..repeats {
            let started = Instant::now();
            let image = pipeline.generate(&request).await?;
            let path = if repeats == 1 {
                PathBuf::from(&args[1])
            } else {
                let path = PathBuf::from(&args[1]);
                path.with_file_name(format!(
                    "{}-{index}.png",
                    path.file_stem().unwrap_or_default().to_string_lossy()
                ))
            };
            image.save_png(&path)?;
            eprintln!(
                "render {index}: {:.3}s -> {}",
                started.elapsed().as_secs_f64(),
                path.display()
            );
        }
        pipeline.unload().await
    })?;
    Ok(())
}
