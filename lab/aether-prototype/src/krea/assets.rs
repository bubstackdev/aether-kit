use super::prompt::SYSTEM;
use crate::{Error, Result, error::require};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
};

/// An offline export registered against the exact original source streams.
/// Paths in a JSON registration resolve relative to that JSON file.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShapeSpecialization {
    pub image_tokens: usize,
    pub text_rows: usize,
    pub source_sha256: [String; 4],
    pub specialized_sha256: [String; 4],
    pub segments: [PathBuf; 4],
}
impl ShapeSpecialization {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut spec: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        for segment in &mut spec.segments {
            *segment = path.parent().unwrap_or(Path::new(".")).join(&*segment);
        }
        Ok(spec)
    }
}

/// Validated Krea-family assets. Weights remain in the external forge bundle.
/// One pipeline retains at most one denoiser set, even with multiple exports.
#[derive(Debug, Clone)]
pub struct KreaAssets {
    pub(crate) encoder: PathBuf,
    pub(crate) decoder: PathBuf,
    pub(crate) tokenizer: PathBuf,
    pub(crate) dynamic: [PathBuf; 4],
    source: [String; 4],
    specializations: Vec<ShapeSpecialization>,
}

impl KreaAssets {
    /// Open the original dynamic `macos-any-gpu` bundle, not the lab's
    /// fixed-shape variant. This guarantees a fallback for other text lengths.
    pub fn open(bundle: impl AsRef<Path>) -> Result<Self> {
        let bundle = bundle.as_ref();
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(bundle.join("metadata.json"))?)?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(bundle.join("aether.json"))?)?;
        let d = &manifest["diffusion"];
        let supported = manifest["schema"] == 1
            && manifest["driver"] == "diffusion"
            && metadata["name"] == "krea_2_turbo_realism"
            && d["latentChannels"] == 64
            && d["pixelsPerToken"] == 16
            && d["sigmaShift"]["rule"] == "constant"
            && d["sigmaShift"]["mu"] == 1.15
            && d["rotary"]["axes"] == serde_json::json!([32, 48, 48])
            && d["rotary"]["theta"] == 1000
            && d["rotary"]["layout"] == "tyx"
            && d["rotary"]["textRotated"] == false
            && (d["trainTimesteps"].is_null() || d["trainTimesteps"] == 1000)
            && d["denoiserSegments"]
                == serde_json::json!(["denoiser0", "denoiser1", "denoiser2", "denoiser3"])
            && d["textEncoderAsset"] == "textEncoder"
            && d["decoderAsset"] == "vaeDecoder"
            && d["guidance"] == false
            && d["steps"] == 8
            && d["prompt"]["systemMessage"] == SYSTEM
            && d["prompt"]["template"] == "chat"
            && d["prompt"]["dropLeadingRows"] == 34
            && d["prompt"]["maxLength"] == 546
            && d["prompt"]["keepTrailingTokens"] == 5
            && d["prompt"]["padding"] == "none"
            && d["prompt"]["thinking"].is_null()
            && manifest["text"]["tokenizer"] == "tokenizer/";
        if !supported {
            return Err(Error::Unsupported(
                "expected the Krea 2 Turbo realism recipe (64 channels, mu 1.15, Qwen prompt)"
                    .into(),
            ));
        }
        let map: BTreeMap<String, PathBuf> = serde_json::from_value(metadata["assets"].clone())?;
        let asset = |key: &str| -> Result<PathBuf> {
            let path = bundle.join(
                map.get(key)
                    .ok_or_else(|| Error::Invalid(format!("missing asset {key}")))?,
            );
            require(
                path.join("main.mlirb").is_file(),
                format!("asset missing: {}", path.display()),
            )?;
            Ok(path)
        };
        let dynamic = [
            asset("denoiser0")?,
            asset("denoiser1")?,
            asset("denoiser2")?,
            asset("denoiser3")?,
        ];
        let source = [
            declared_hash(&dynamic[0])?,
            declared_hash(&dynamic[1])?,
            declared_hash(&dynamic[2])?,
            declared_hash(&dynamic[3])?,
        ];
        let tokenizer = bundle.join("tokenizer/tokenizer.json");
        require(tokenizer.is_file(), "missing tokenizer/tokenizer.json")?;
        Ok(Self {
            encoder: asset("textEncoder")?,
            decoder: asset("vaeDecoder")?,
            tokenizer,
            dynamic,
            source,
            specializations: Vec::new(),
        })
    }

    /// Register a trusted offline export. Both source and specialized bytecode
    /// are streamed through SHA-256 before registration; never select a shape
    /// export using a model name or dimensions alone. Call during setup.
    pub fn add_specialization(&mut self, spec: ShapeSpecialization) -> Result<()> {
        require(
            self.specializations.len() < 8,
            "at most eight offline shape registrations",
        )?;
        require(
            spec.image_tokens > 0
                && spec.image_tokens <= 4096
                && (1..=512).contains(&spec.text_rows),
            "invalid specialization shape",
        )?;
        require(
            spec.source_sha256 == self.source,
            "specialization belongs to different source weights",
        )?;
        require(
            !self
                .specializations
                .iter()
                .any(|s| s.image_tokens == spec.image_tokens && s.text_rows == spec.text_rows),
            "duplicate specialization shape",
        )?;
        for i in 0..4 {
            verify_asset(&self.dynamic[i], &spec.source_sha256[i])?;
            verify_asset(&spec.segments[i], &spec.specialized_sha256[i])?;
        }
        self.specializations.push(spec);
        Ok(())
    }
    /// Verify all dynamic model streams, including encoder and decoder.
    pub fn verify(&self) -> Result<()> {
        for path in self.dynamic.iter().chain([&self.encoder, &self.decoder]) {
            verify_asset(path, &declared_hash(path)?)?;
        }
        Ok(())
    }
    pub fn has_specialization(&self, image_tokens: usize, text_rows: usize) -> bool {
        self.specializations
            .iter()
            .any(|s| s.image_tokens == image_tokens && s.text_rows == text_rows)
    }
    pub(crate) fn select(&self, image_tokens: usize, text_rows: usize) -> (&[PathBuf; 4], bool) {
        match self
            .specializations
            .iter()
            .find(|s| s.image_tokens == image_tokens && s.text_rows == text_rows)
        {
            Some(s) => (&s.segments, true),
            None => (&self.dynamic, false),
        }
    }
}

fn declared_hash(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path.join("main.hash"))?;
    require(
        bytes.len() == 32,
        format!("invalid main.hash in {}", path.display()),
    )?;
    Ok(format!("{:x}", DigestBytes(&bytes)))
}
struct DigestBytes<'a>(&'a [u8]);
impl std::fmt::LowerHex for DigestBytes<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}
fn verify_asset(path: &Path, expected: &str) -> Result<()> {
    require(
        declared_hash(path)? == expected,
        format!("declared asset hash mismatch: {}", path.display()),
    )?;
    let mut file = std::fs::File::open(path.join("main.mlirb"))?;
    let mut buffer = vec![0; 4 << 20];
    let mut digest = Sha256::new();
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    require(
        format!("{:x}", DigestBytes(&digest.finalize())) == expected,
        format!("asset bytes differ from SHA-256: {}", path.display()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (tempfile::TempDir, KreaAssets, ShapeSpecialization) {
        let temp = tempfile::tempdir().unwrap();
        let mut paths = Vec::new();
        let mut hashes = Vec::new();
        for i in 0..10 {
            let path = temp.path().join(format!("{i}.aimodel"));
            std::fs::create_dir(&path).unwrap();
            let bytes = format!("fake model {i}");
            let hash = Sha256::digest(bytes.as_bytes());
            std::fs::write(path.join("main.mlirb"), &bytes).unwrap();
            std::fs::write(path.join("main.hash"), hash).unwrap();
            paths.push(path);
            hashes.push(format!("{:x}", DigestBytes(&hash)));
        }
        let dynamic: [PathBuf; 4] = paths[..4].to_vec().try_into().unwrap();
        let source: [String; 4] = hashes[..4].to_vec().try_into().unwrap();
        let assets = KreaAssets {
            encoder: paths[8].clone(),
            decoder: paths[9].clone(),
            tokenizer: temp.path().join("tokenizer.json"),
            dynamic,
            source: source.clone(),
            specializations: Vec::new(),
        };
        let spec = ShapeSpecialization {
            image_tokens: 4032,
            text_rows: 20,
            source_sha256: source,
            specialized_sha256: hashes[4..8].to_vec().try_into().unwrap(),
            segments: paths[4..8].to_vec().try_into().unwrap(),
        };
        (temp, assets, spec)
    }

    #[test]
    fn shape_selection_requires_both_dimensions_and_source_identity() {
        let (_temp, mut assets, spec) = fixture();
        assets.add_specialization(spec.clone()).unwrap();
        assert_eq!(assets.select(4032, 20), (&spec.segments, true));
        for (tokens, text) in [(4032, 19), (4096, 20), (1024, 18)] {
            assert_eq!(assets.select(tokens, text), (&assets.dynamic, false));
        }
        assert!(assets.add_specialization(spec.clone()).is_err());
        let mut wrong = spec;
        wrong.text_rows = 19;
        wrong.source_sha256[0] = "00".repeat(32);
        assert!(assets.add_specialization(wrong).is_err());
    }

    #[test]
    fn tampered_bytecode_never_registers_even_with_unchanged_main_hash() {
        let (_temp, mut assets, spec) = fixture();
        std::fs::write(spec.segments[2].join("main.mlirb"), b"tampered").unwrap();
        assert!(assets.add_specialization(spec).is_err());
        assert!(!assets.has_specialization(4032, 20));
        assert!(assets.verify().is_ok());
        std::fs::write(assets.encoder.join("main.mlirb"), b"tampered").unwrap();
        assert!(assets.verify().is_err());
    }
}
