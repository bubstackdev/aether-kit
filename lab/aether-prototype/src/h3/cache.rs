//! Optional application-owned disk cache of completed native preparation.
//! Files are immutable after publication. Hashes detect accidental corruption;
//! the cache directory, like model assets, must be trusted by the application.
use super::{
    H3RowTimesteps,
    assets::{Manifest, TensorFile, hash},
};
use crate::{Result, error::require};
use corsairs::{DType, Tensor};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub(crate) struct Prepared {
    pub embeddings: Vec<Tensor>,
    pub tables: Vec<Vec<Tensor>>,
}
#[derive(Deserialize, Serialize)]
struct Record {
    schema: usize,
    key: String,
    embeddings: Vec<TensorFile>,
    tables: Vec<Vec<TensorFile>>,
}
pub(crate) fn key(m: &Manifest, plans: &[H3RowTimesteps]) -> Result<String> {
    // Include the entire schedule, even when two current plans share values.
    // No prompt/seed is needed: these functions consume only exact timesteps.
    let value = serde_json::json!({"schema":1,"runtime":"corsairs-0.3.0", "recipe":m.recipe,
        "adapter":m.adapter, "schedule_sha256":hash(&std::fs::read(&m.schedule)?),
        "time_model":{"sha256":m.time.sha256,"function":m.time.function},
        "modulation_models":m.modulations.iter().map(|a| serde_json::json!({"sha256":a.sha256,"function":a.function})).collect::<Vec<_>>(),
        "timestep_bits":plans.iter().map(|p| p.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>()});
    Ok(hash(&serde_json::to_vec(&value)?))
}
impl Prepared {
    fn validate(&self, plans: &[H3RowTimesteps]) -> Result<()> {
        require(
            self.embeddings.len() == plans.len()
                && self.tables.len() == 50
                && self.tables.iter().all(|b| b.len() == plans.len()),
            "H3 preparation cache lengths differ",
        )?;
        for (i, plan) in plans.iter().enumerate() {
            let e = &self.embeddings[i];
            require(
                e.dtype() == DType::Float32 && e.shape() == [plan.values.len(), 2688],
                "H3 cached time embedding differs",
            )?;
            for block in &self.tables {
                let t = &block[i];
                require(
                    t.dtype() == DType::BFloat16 && t.shape() == [6, plan.values.len() * 3, 5376],
                    "H3 cached modulation shape differs",
                )?;
            }
        }
        Ok(())
    }
    pub fn load(root: &Path, key: &str, plans: &[H3RowTimesteps]) -> Result<Option<Self>> {
        let directory = root.join(key);
        let bytes = match std::fs::read(directory.join("cache.json")) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && !directory.exists() => {
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };
        require(
            bytes.len() <= 4 << 20,
            "H3 preparation cache record exceeds limit",
        )?;
        let record: Record = serde_json::from_slice(&bytes)?;
        require(
            record.schema == 1 && record.key == key,
            "H3 preparation cache identity differs",
        )?;
        let read = |mut t: TensorFile| {
            require(
                t.path.components().count() == 1
                    && t.path
                        .file_name()
                        .is_some_and(|n| n.to_string_lossy().ends_with(".raw")),
                "H3 cache tensor path must be a filename",
            )?;
            t.path = directory.join(t.path);
            t.read()
        };
        let prepared = Self {
            embeddings: record
                .embeddings
                .into_iter()
                .map(&read)
                .collect::<Result<_>>()?,
            tables: record
                .tables
                .into_iter()
                .map(|b| b.into_iter().map(&read).collect::<Result<_>>())
                .collect::<Result<_>>()?,
        };
        prepared.validate(plans)?;
        Ok(Some(prepared))
    }
    pub fn save(&self, root: &Path, key: &str, plans: &[H3RowTimesteps]) -> Result<()> {
        self.validate(plans)?;
        std::fs::create_dir_all(root)?;
        let directory = root.join(key);
        // Another pipeline may have published an identical entry while we ran.
        if directory.exists() {
            Self::load(root, key, plans)?;
            return Ok(());
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| crate::Error::Invalid("system clock precedes epoch".into()))?
            .as_nanos();
        let temporary = root.join(format!(".{key}-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&temporary)?;
        let result = (|| {
            let write = |name: String, tensor: &Tensor| -> Result<TensorFile> {
                let filename = format!("{name}.raw");
                std::fs::write(temporary.join(&filename), tensor.as_bytes())?;
                Ok(TensorFile {
                    name,
                    path: filename.into(),
                    dtype: match tensor.dtype() {
                        DType::Float32 => "float32",
                        _ => "bfloat16",
                    }
                    .into(),
                    shape: tensor.shape().to_vec(),
                    sha256: hash(tensor.as_bytes()),
                })
            };
            let embeddings = self
                .embeddings
                .iter()
                .enumerate()
                .map(|(i, t)| write(format!("time-{i}"), t))
                .collect::<Result<_>>()?;
            let tables = self
                .tables
                .iter()
                .enumerate()
                .map(|(b, values)| {
                    values
                        .iter()
                        .enumerate()
                        .map(|(i, t)| write(format!("block-{b}-step-{i}"), t))
                        .collect::<Result<_>>()
                })
                .collect::<Result<_>>()?;
            std::fs::write(
                temporary.join("cache.json"),
                serde_json::to_vec(&Record {
                    schema: 1,
                    key: key.into(),
                    embeddings,
                    tables,
                })?,
            )?;
            match std::fs::rename(&temporary, &directory) {
                Ok(()) => Ok(()),
                Err(_) if directory.exists() => {
                    Self::load(root, key, plans)?;
                    Ok(())
                }
                Err(e) => Err(e.into()),
            }
        })();
        if temporary.exists() {
            std::fs::remove_dir_all(temporary)?;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_identity_covers_models_adapter_schedule_and_exact_timestep_bits() {
        let temp = tempfile::tempdir().unwrap();
        let schedule = temp.path().join("schedule.json");
        std::fs::write(&schedule, b"original").unwrap();
        let model = serde_json::json!({"path":"unused","function":"main","sha256":"a".repeat(64)});
        let tensor = serde_json::json!({"name":"unused","path":"unused","dtype":"bfloat16","shape":[1],"sha256":"b".repeat(64)});
        let mut m:Manifest=serde_json::from_value(serde_json::json!({"schema":1,"complete":true,"snapshot":"unused","diffusers_commit":"unused",
            "width":960,"height":544,"frames":124,"text_rows":49,"tokenizer":"unused","schedule":schedule,
            "prepare":model,"time":model,"heads":model,"bodies":[],"modulations":vec![model.clone();50],
            "parameters":[],"rotary_cos":tensor,"rotary_sin":tensor,"video_decoder":model,"audio_decoder":model})).unwrap();
        let mut plans = vec![H3RowTimesteps {
            values: vec![0.0],
            indices: vec![],
            adaln_indices: vec![],
        }];
        let initial = key(&m, &plans).unwrap();
        m.time.sha256 = "c".repeat(64);
        assert_ne!(initial, key(&m, &plans).unwrap());
        m.time.sha256 = "a".repeat(64);
        m.modulations[23].sha256 = "c".repeat(64);
        assert_ne!(initial, key(&m, &plans).unwrap());
        m.modulations[23].sha256 = "a".repeat(64);
        plans[0].values[0] = -0.0;
        assert_ne!(initial, key(&m, &plans).unwrap());
        plans[0].values[0] = 0.0;
        std::fs::write(&schedule, b"changed").unwrap();
        assert_ne!(initial, key(&m, &plans).unwrap());
        std::fs::write(&schedule, b"original").unwrap();
        m.adapter = Some(super::super::assets::AdapterIdentity {
            repository: "adapter".into(),
            revision: "revision".into(),
            sha256: "c".repeat(64),
            execution: "dynamic_lora".into(),
            rank: 128,
            alpha: 8.0,
        });
        assert_ne!(initial, key(&m, &plans).unwrap());
    }
    #[test]
    fn prepared_cache_roundtrip_rejects_corruption_and_wrong_cardinality() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let plans = vec![H3RowTimesteps {
            values: vec![0.0],
            indices: vec![0],
            adaln_indices: vec![0],
        }];
        let t =
            Tensor::from_bytes(DType::BFloat16, [6, 3, 5376], vec![0; 6 * 3 * 5376 * 2]).unwrap();
        let prepared = Prepared {
            embeddings: vec![Tensor::from_vec([1, 2688], vec![0.0f32; 2688]).unwrap()],
            tables: vec![vec![t]; 50],
        };
        assert!(Prepared::load(root, "missing", &plans).unwrap().is_none());
        prepared.save(root, "test", &plans).unwrap();
        assert_eq!(
            Prepared::load(root, "test", &plans)
                .unwrap()
                .unwrap()
                .tables
                .len(),
            50
        );
        let mut changed = plans.clone();
        changed[0].values.push(0.5);
        assert!(Prepared::load(root, "test", &changed).is_err());
        std::fs::write(
            root.join("test/block-0-step-0.raw"),
            vec![0; 6 * 3 * 5376 * 2 - 1],
        )
        .unwrap();
        assert!(Prepared::load(root, "test", &plans).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
