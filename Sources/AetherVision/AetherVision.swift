// AetherVision: the vision drivers (DESIGN.md §8). So far `depth-estimator@1` (``DepthEstimator``, M4b) and
// `detector-dense@1` (``ObjectDetector``, M4e); CLIP and super-resolution follow. Their image input and preprocessing (`ImageInput`, `ImagePreprocessor`,
// `PreparedImage`) are in AetherRuntime and their shared driver core (`StatelessPipeline`) in AetherHost, both
// re-exported here.
@_exported public import AetherHost
