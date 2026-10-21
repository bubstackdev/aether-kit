use crate::{Error, Result, error::require};
use std::path::Path;
use tokenizers::Tokenizer;

pub(crate) const SYSTEM: &str = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:";

/// Fully rendered, unpadded Krea encoder input. Selected rows start at row 34.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KreaPrompt {
    ids: Vec<i32>,
    truncated: bool,
}
impl KreaPrompt {
    pub fn token_ids(&self) -> &[i32] {
        &self.ids
    }
    pub fn text_rows(&self) -> usize {
        self.ids.len() - 34
    }
    pub fn truncated(&self) -> bool {
        self.truncated
    }
    /// For fixture replay or callers supplying the exact Krea chat rendering.
    pub fn from_token_ids(ids: Vec<i32>) -> Result<Self> {
        require(
            (35..=546).contains(&ids.len()) && ids.iter().all(|i| *i >= 0),
            "Krea needs 35..=546 nonnegative token IDs",
        )?;
        Ok(Self {
            ids,
            truncated: false,
        })
    }
}

pub struct KreaTokenizer(Tokenizer);
impl KreaTokenizer {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut tokenizer =
            Tokenizer::from_file(path).map_err(|e| Error::Tokenizer(e.to_string()))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(Self(tokenizer))
    }
    /// The text-only branch of Krea's Qwen chat template. Other model families
    /// supply their own template rather than sharing this rendering.
    pub fn prepare(&self, prompt: &str) -> Result<KreaPrompt> {
        let rendered = format!(
            "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let encoding = self
            .0
            .encode(rendered, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut ids = encoding
            .get_ids()
            .iter()
            .map(|i| {
                i32::try_from(*i).map_err(|_| Error::Tokenizer("token ID exceeds int32".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let truncated = ids.len() > 546;
        if truncated {
            let suffix = ids[ids.len() - 5..].to_vec();
            ids.truncate(541);
            ids.extend(suffix);
        }
        let mut result = KreaPrompt::from_token_ids(ids)?;
        result.truncated = truncated;
        Ok(result)
    }
}
