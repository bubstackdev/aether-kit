use super::*;

const ROWS: usize = 19350;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SourceBlock {
    block: usize,
    #[serde(rename = "sourceTensorSHA256")]
    source_sha256: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportRecord {
    schema_version: usize,
    snapshot: String,
    diffusers_commit: String,
    rows: usize,
    source_dtype_preserved: bool,
    completed: bool,
    blocks: Vec<SourceBlock>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SegmentRecord {
    segment: usize,
    first_block: usize,
    end_block_exclusive: usize,
    #[serde(rename = "sourceBlockSHA256")]
    source_block_sha256: Vec<String>,
    asset: PathBuf,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SegmentsRecord {
    schema_version: usize,
    snapshot: String,
    rows: usize,
    completed: bool,
    blocks_per_segment: usize,
    segments: Vec<SegmentRecord>,
}
#[derive(Deserialize)]
struct TableRecord {
    block: usize,
    dtype: String,
    shape: Vec<usize>,
    file: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TablesRecord {
    schema_version: usize,
    #[serde(rename = "sourceExportRecordSHA256")]
    source_export_record_sha256: String,
    #[serde(rename = "tembSHA256")]
    temb_sha256: String,
    files: Vec<TableRecord>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BackboneReference {
    schema_version: usize,
    status: String,
    rows: usize,
    blocks_completed: usize,
    blocks_expected: usize,
    final_finite: bool,
    #[serde(rename = "finalSHA256")]
    final_sha256: String,
}
struct Plan {
    oracle: Oracle,
    segments: Vec<SegmentRecord>,
    tables: Vec<Fixture>,
    reference: Fixture,
    records_sha256: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct H3BackboneSegment {
    pub segment: usize,
    pub first_block: usize,
    pub end_block_exclusive: usize,
    pub model_source_sha256: String,
    pub output_sha256: String,
    pub finite: bool,
    pub drain_completed: bool,
}
#[derive(Debug, Serialize)]
pub struct H3BackboneQualification {
    pub passed: bool,
    pub scope: String,
    pub rows: usize,
    pub native_calls: usize,
    pub segments: Vec<H3BackboneSegment>,
    pub output_sha256: String,
    pub expected_output_sha256: String,
    pub byte_identical_to_swift: bool,
    pub records_sha256: BTreeMap<String, String>,
}

impl H3Lab {
    fn backbone_plan(&self) -> Result<Plan> {
        let names = [
            "backbone-export.json",
            "segments-25-export.json",
            "backbone-tables.json",
            "backbone-streaming.json",
            "block-oracle-19350.json",
        ];
        let records: BTreeMap<_, _> = names
            .iter()
            .map(|name| Ok((*name, std::fs::read(self.root.join("records").join(name))?)))
            .collect::<Result<_>>()?;
        let export: ExportRecord = serde_json::from_slice(&records[names[0]])?;
        let segments: SegmentsRecord = serde_json::from_slice(&records[names[1]])?;
        let tables: TablesRecord = serde_json::from_slice(&records[names[2]])?;
        let reference: BackboneReference = serde_json::from_slice(&records[names[3]])?;
        let oracle = self.oracle(ROWS)?;
        require(
            export.schema_version == 1
                && export.snapshot == SNAPSHOT
                && export.diffusers_commit == DIFFUSERS_COMMIT
                && export.rows == ROWS
                && export.source_dtype_preserved
                && export.completed
                && export.blocks.len() == 50,
            "H3 backbone export identity differs",
        )?;
        require(
            segments.schema_version == 1
                && segments.snapshot == SNAPSHOT
                && segments.rows == ROWS
                && segments.completed
                && segments.blocks_per_segment == 25
                && segments.segments.len() == 2,
            "H3 backbone requires the two retained original 25-block segments",
        )?;
        for (index, segment) in segments.segments.iter().enumerate() {
            require(
                segment.segment == index
                    && segment.first_block == index * 25
                    && segment.end_block_exclusive == (index + 1) * 25
                    && segment.asset
                        == Path::new(&format!("forge/assets/segments-25/segment-{index}.aimodel"))
                    && segment.source_block_sha256.len() == 25,
                "H3 segment order, range or asset path differs",
            )?;
            for (offset, digest) in segment.source_block_sha256.iter().enumerate() {
                let block = &export.blocks[index * 25 + offset];
                require(
                    block.block == index * 25 + offset
                        && block.source_sha256 == *digest
                        && digest.len() == 64,
                    "H3 segment source blocks differ from backbone export",
                )?;
            }
        }
        require(
            tables.schema_version == 1
                && tables.source_export_record_sha256 == hash(&records[names[0]])
                && tables.files.len() == 50
                && oracle
                    .inputs
                    .iter()
                    .any(|f| f.name == "temb" && f.sha256 == tables.temb_sha256),
            "H3 modulation tables belong to a different export or timestep pair",
        )?;
        let tables = tables
            .files
            .into_iter()
            .enumerate()
            .map(|(index, table)| {
                require(
                    table.block == index
                        && table.dtype == "bfloat16"
                        && table.shape == [6, 9, 5376],
                    "H3 backbone table order or signature differs",
                )?;
                Ok(Fixture {
                    name: format!("block-{index}-modulations"),
                    dtype: table.dtype,
                    shape: table.shape,
                    file: table.file,
                    sha256: table.sha256,
                })
            })
            .collect::<Result<_>>()?;
        require(
            reference.schema_version == 1
                && reference.rows == ROWS
                && reference.status == "completed-backbone-pass"
                && reference.blocks_completed == 50
                && reference.blocks_expected == 50
                && reference.final_finite
                && reference.final_sha256.len() == 64,
            "H3 Swift backbone reference is incomplete",
        )?;
        Ok(Plan {
            oracle,
            segments: segments.segments,
            tables,
            reference: Fixture {
                name: "Swift backbone output".into(),
                dtype: "bfloat16".into(),
                shape: vec![1, ROWS, 5376],
                file: "forge/fixtures/backbone-streaming.raw".into(),
                sha256: reference.final_sha256,
            },
            records_sha256: records
                .iter()
                .map(|(name, bytes)| (format!("records/{name}"), hash(bytes)))
                .collect(),
        })
    }

    /// Hash the fixed synthetic backbone inputs, 50 modulation tables and Swift
    /// output. Checks export provenance without hashing the 38.5 GB model streams
    /// or creating a runtime; `qualify_backbone` verifies those streams too.
    pub fn inspect_backbone(&self) -> Result<serde_json::Value> {
        let plan = self.backbone_plan()?;
        for fixture in plan
            .oracle
            .inputs
            .iter()
            .chain(&plan.tables)
            .chain([&plan.reference])
        {
            self.read(fixture)?.lower_indices()?;
        }
        let missing: Vec<_> = plan
            .segments
            .iter()
            .filter(|s| !self.root.join(&s.asset).join("main.mlirb").is_file())
            .map(|s| &s.asset)
            .collect();
        Ok(serde_json::json!({
            "scope":"fixed synthetic 50-block backbone inputs and reference; no native execution",
            "rows":ROWS,"blocks":50,"segments":2,"tables":50,
            "records_sha256":plan.records_sha256,
            "expected_output_sha256":plan.reference.sha256,
            "model_streams_hashed":false,"missing_segment_assets":missing,
            "clip_generation_implemented":false
        }))
    }

    /// Run the retained synthetic 50-block backbone in one runtime. One graph
    /// is acquired at a time, with an owned BF16 host carry and a waited drain
    /// before the next acquisition. This does not establish repeated residency,
    /// OS memory eviction or a complete H3 generation step.
    pub async fn qualify_backbone(&self, runtime: &Runtime) -> Result<H3BackboneQualification> {
        let result = self.run_backbone(runtime).await;
        // Error paths also drop stage handles before transitive cleanup.
        runtime.drain().await?;
        result
    }

    async fn run_backbone(&self, runtime: &Runtime) -> Result<H3BackboneQualification> {
        let plan = self.backbone_plan()?;
        self.inspect_backbone()?;
        let model_hashes = plan
            .segments
            .iter()
            .map(|segment| verify_asset(&self.root.join(&segment.asset)))
            .collect::<Result<Vec<_>>>()?;
        let mut base = BTreeMap::new();
        for fixture in &plan.oracle.inputs {
            if fixture.name != "temb" {
                require(
                    base.insert(fixture.name.clone(), self.read(fixture)?.lower_indices()?)
                        .is_none(),
                    "duplicate H3 backbone input",
                )?;
            }
        }
        let initial = base
            .remove("hidden")
            .ok_or_else(|| Error::Invalid("H3 backbone hidden input missing".into()))?;
        require(
            initial.dtype == "bfloat16" && initial.shape == [1, ROWS, 5376],
            "H3 backbone hidden signature differs",
        )?;
        let mut hidden =
            Tensor::from_bytes(corsairs::DType::BFloat16, initial.shape, initial.bytes)?;
        let mut reports = Vec::new();
        for segment in &plan.segments {
            let mut tables = Vec::with_capacity(25 * 6 * 9 * 5376 * 2);
            for fixture in &plan.tables[segment.first_block..segment.end_block_exclusive] {
                tables.extend_from_slice(&self.read(fixture)?.bytes);
            }
            let model = runtime
                .load_model_with_options(
                    self.root.join(&segment.asset),
                    LoadOptions::new(Compute::GpuPreferred).with_expect_frequent_reshapes(true),
                )
                .await
                .map_err(|source| Error::Stage {
                    stage: format!("H3 backbone segment {} model load", segment.segment),
                    source,
                })?;
            let function = model
                .load_function("segment")
                .await
                .map_err(|source| Error::Stage {
                    stage: format!("H3 backbone segment {} function load", segment.segment),
                    source,
                })?;
            drop(model);
            require(
                function.descriptor().states.is_empty()
                    && function.descriptor().inputs.len() == base.len() + 2,
                "H3 backbone input/state contract differs",
            )?;
            let mut inputs = TensorMap::from([
                ("hidden", hidden),
                (
                    "tables",
                    Tensor::from_bytes(corsairs::DType::BFloat16, [25, 6, 9, 5376], tables)?,
                ),
            ]);
            for (name, payload) in &base {
                let descriptor = function
                    .descriptor()
                    .inputs
                    .iter()
                    .find(|d| d.name == *name)
                    .ok_or_else(|| Error::Invalid(format!("H3 segment omitted {name}")))?;
                inputs.insert(name, payload.to_native(descriptor)?);
            }
            let result = function.run(inputs, TensorMap::new()).await;
            drop(function);
            runtime.drain().await?;
            let mut result = result?;
            require(
                result.states.is_empty() && result.outputs.len() == 1,
                "unexpected H3 segment outputs or states",
            )?;
            hidden = result
                .outputs
                .remove("hidden_out")
                .ok_or_else(|| Error::Invalid("H3 segment output missing".into()))?;
            check_output(&hidden, ROWS)?;
            let finite = hidden.as_bytes().as_chunks::<2>().0.iter().all(|bits| {
                corsairs::BFloat16::from_bits(u16::from_le_bytes(*bits))
                    .to_f32()
                    .is_finite()
            });
            reports.push(H3BackboneSegment {
                segment: segment.segment,
                first_block: segment.first_block,
                end_block_exclusive: segment.end_block_exclusive,
                model_source_sha256: model_hashes[segment.segment].clone(),
                output_sha256: hash(hidden.as_bytes()),
                finite,
                drain_completed: true,
            });
            require(finite, "nonfinite H3 backbone hidden state")?;
        }
        let output_sha256 = hash(hidden.as_bytes());
        let byte_identical_to_swift = output_sha256 == plan.reference.sha256
            && hidden.as_bytes() == self.read(&plan.reference)?.bytes;
        Ok(H3BackboneQualification {
            passed: byte_identical_to_swift,
            scope: "one fixed synthetic 50-block hidden-state pass at one timestep pair; two serial 25-block graphs in one Runtime with owned BF16 carry and transitive drain; excludes conditioning, packing, heads, latent updates and decode".into(),
            rows: ROWS,
            native_calls: reports.len(),
            segments: reports,
            output_sha256,
            expected_output_sha256: plan.reference.sha256,
            byte_identical_to_swift,
            records_sha256: plan.records_sha256,
        })
    }
}
