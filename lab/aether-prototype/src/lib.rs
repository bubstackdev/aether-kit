//! Generative pipelines on Corsairs. Start with [`krea::KreaPipeline`].
//!
//! Model families own their conditioning and sampling semantics. Modality APIs
//! such as [`ImagePipeline`] and [`AudioVideoPipeline`] own requests and artifacts;
//! Corsairs owns execution. [`h3::H3Pipeline`] orchestrates H3 text-to-audio/video
//! with offline Forge exports; [`h3::H3Lab`] retains bounded lab qualification.
//! There is no process-global runtime, GPU state, or network weight download.
#![forbid(unsafe_code)]

mod audio_video;
mod error;
pub mod h3;
mod image;
pub mod krea;

pub use audio_video::{
    AudioVideoPipeline, AudioVideoRequest, AudioVideoWriter, GeneratedAudioVideo, PcmAudio,
};
pub use error::{Error, Result};
pub use image::{GeneratedImage, ImagePipeline, ImageRequest, ImageSize};
