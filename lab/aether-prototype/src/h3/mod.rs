//! MiniMax H3 T2VA export contracts, generation and bounded benchmarking.
//! The separate lab adapter retains original block/backbone qualification.
mod assets;
mod cache;
mod contract;
mod decode;
pub use decode::H3VideoStitcher;
mod lab;
mod layout;
mod pipeline;
mod spatial;
pub use assets::H3Assets;
pub use contract::{H3Geometry, H3Recipe, H3Schedule, H3Step};
pub use lab::{
    H3BackboneQualification, H3BackboneSegment, H3Lab, H3Qualification, H3ScheduleQualification,
    H3ScheduleSample,
};
pub use layout::{H3Anchor, H3Layout, H3RowTimesteps, pack_video};
pub use pipeline::{H3InitialLatents, H3Options, H3Pipeline, H3Progress};
pub use spatial::{H3SpatialStitcher, H3SpatialTiling};

pub const SNAPSHOT: &str = "42ed227ee7df40d41602854ae760620d6eb651fe";
pub const DIFFUSERS_COMMIT: &str = "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941";
