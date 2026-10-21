use aether::{
    AudioVideoRequest, AudioVideoWriter, ImageSize, PcmAudio,
    h3::{H3Anchor, H3Geometry, H3Lab, H3Layout, H3Recipe, H3Schedule, pack_video},
};

#[test]
fn h3_geometry_tracks_both_frame_grids_and_joint_video_rows() {
    let mut request = AudioVideoRequest::new(
        "a bird with rustling wings",
        ImageSize::new(960, 544),
        124,
        42,
    );
    let geometry = H3Geometry::for_request(&request, false).unwrap();
    assert_eq!(
        (
            geometry.frames,
            geometry.latent_frames,
            geometry.target_video_rows
        ),
        (124, 37, 18870)
    );
    request.size = ImageSize::new(1344, 768);
    assert_eq!(
        H3Geometry::for_request(&request, false)
            .unwrap()
            .target_video_rows,
        37296
    );
    request.first_frame = Some("first.png".into());
    request.last_frame = Some("last.png".into());
    assert_eq!(
        H3Geometry::for_request(&request, false).unwrap().keyframes,
        2
    );
    request.frames = 120;
    assert_eq!(
        H3Geometry::for_request(&request, false).unwrap().frames,
        124
    );
    request.frames = 22;
    assert!(H3Geometry::for_request(&request, false).is_err());
    assert_eq!(
        H3Geometry::for_request(&request, true)
            .unwrap()
            .latent_frames,
        7
    );
    request.frames = 360;
    assert!(H3Geometry::for_request(&request, false).is_err());
    request.frames = 345;
    let long = H3Geometry::for_request(&request, false).unwrap();
    assert_eq!(long.target_video_rows, 102816);
    let hidden_bytes = long.target_video_rows * 5376 * 2;
    assert_eq!(hidden_bytes, 1_105_477_632);
    assert!(hidden_bytes > 1 << 30);
    request.frames = 124;
    request.size = ImageSize::new(usize::MAX, 544);
    assert!(H3Geometry::for_request(&request, false).is_err());
}

#[test]
fn invalid_schedule_is_rejected_before_execution() {
    let mut record = serde_json::json!({"gridPoints":51,"expectedForwards":50,
        "schedules":[{"shift":12.0,"timesteps":vec![0.0;50],"sigmas":vec![1.0;51]},
                     {"shift":3.0,"timesteps":vec![0.0;50],"sigmas":vec![1.0;51]}]});
    assert!(H3Schedule::from_json(&serde_json::to_vec(&record).unwrap()).is_err());
    record["expectedForwards"] = 49.into();
    assert!(H3Schedule::from_json(&serde_json::to_vec(&record).unwrap()).is_err());
}

#[test]
fn turbo_schedule_requires_its_recipe_and_matches_all_eight_upstream_updates() {
    use sha2::{Digest, Sha256};
    let bytes = include_bytes!("../src/h3/recipes/turbo8-schedule.json");
    assert!(H3Schedule::from_json(bytes).is_err());
    assert!(
        H3Schedule::from_json_for_recipe(
            include_bytes!("fixtures/h3-schedule.json"),
            H3Recipe::Turbo8
        )
        .is_err()
    );
    let schedule = H3Schedule::from_json_for_recipe(bytes, H3Recipe::Turbo8).unwrap();
    let mut wrong: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    wrong["schedules"][0]["sigmas"][1] = 0.99.into();
    wrong["schedules"][0]["timesteps"][1] = (1.0f32 - 0.99).into();
    assert!(
        H3Schedule::from_json_for_recipe(&serde_json::to_vec(&wrong).unwrap(), H3Recipe::Turbo8)
            .is_err()
    );
    let steps: Vec<_> = schedule.steps().collect();
    assert_eq!(steps.len(), 8);
    let request = AudioVideoRequest::new("", ImageSize::new(960, 544), 124, 0);
    let layout = H3Layout::new(
        H3Geometry::for_request(&request, false).unwrap(),
        &[1; 49],
        &[],
    )
    .unwrap();
    let record: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/h3-turbo-numerics.json")).unwrap();
    let floats = |v: &serde_json::Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>()
    };
    let hash = |values: &[f32]| {
        Sha256::digest(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
    };
    let mut video = floats(&record["video"]);
    let mut audio = floats(&record["audio"]);
    for (i, step) in steps.iter().enumerate() {
        let plan = layout.timesteps(*step);
        assert_eq!(plan.values.len(), if i == 0 { 1 } else { 2 });
        for (&index, &tag) in plan.adaln_indices.iter().zip(&layout.token_tags) {
            assert_eq!(index, tag + if i > 0 && tag == 2 { 3 } else { 0 });
        }
        let expected = &record["steps"][i];
        schedule
            .advance_video(i, &mut video, &floats(&expected["video_velocity"]))
            .unwrap();
        schedule
            .advance_audio(i, &mut audio, &floats(&expected["audio_velocity"]))
            .unwrap();
        assert_eq!(hash(&video), expected["video_sha256"]);
        assert_eq!(hash(&audio), expected["audio_sha256"]);
    }
    assert!(schedule.advance_video(8, &mut video, &[0.0; 32]).is_err());
}

#[test]
#[ignore = "requires AETHER_H3_TURBO_BUNDLE; validates CPU export recipe and rejects incompatible metadata, no native execution"]
fn turbo_bundle_pins_dynamic_adapter_and_all_fifty_binding_maps() {
    use aether::h3::H3Assets;
    let path = std::path::PathBuf::from(
        std::env::var_os("AETHER_H3_TURBO_BUNDLE").expect("set AETHER_H3_TURBO_BUNDLE"),
    );
    let assets = H3Assets::open(&path).unwrap();
    assert_eq!(assets.recipe(), H3Recipe::Turbo8);
    assert_eq!(assets.geometry().unwrap().frames, 124);
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let changed = temporary.path().join("bundle.json");
    for layout in [
        serde_json::Value::Null,
        serde_json::json!("unconverted_raw_qkv"),
    ] {
        let mut bad = original.clone();
        bad["transformer_weight_layout"] = layout;
        std::fs::write(&changed, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(H3Assets::open(&changed).is_err());
    }
    let mut bad = original.clone();
    bad.as_object_mut()
        .unwrap()
        .remove("transformer_weight_layout");
    std::fs::write(&changed, serde_json::to_vec(&bad).unwrap()).unwrap();
    assert!(H3Assets::open(&changed).is_err());
    for (field, value) in [
        ("execution", serde_json::json!("merged_bf16")),
        ("sha256", serde_json::json!("0".repeat(64))),
        ("alpha", serde_json::json!(1.0)),
    ] {
        let mut bad = original.clone();
        bad["adapter"][field] = value;
        std::fs::write(&changed, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(H3Assets::open(&changed).is_err());
    }
    let mut bad = original.clone();
    bad["recipe"] = "base50".into();
    std::fs::write(&changed, serde_json::to_vec(&bad).unwrap()).unwrap();
    assert!(H3Assets::open(&changed).is_err());
    let mut bad = original;
    bad["parameters"][49][21]["name"] = "w0".into();
    std::fs::write(&changed, serde_json::to_vec(&bad).unwrap()).unwrap();
    assert!(H3Assets::open(&changed).is_err());
}

#[test]
#[ignore = "requires AETHER_H3_FUNCTIONAL_BUNDLE; checks actual explicit VAE tensor extents without native inference"]
fn functional_video_bundle_preserves_all_original_f32_parameter_extents() {
    use aether::h3::H3Assets;
    let path = std::path::PathBuf::from(
        std::env::var_os("AETHER_H3_FUNCTIONAL_BUNDLE").expect("set AETHER_H3_FUNCTIONAL_BUNDLE"),
    );
    let assets = H3Assets::open(&path).unwrap();
    assert_eq!(assets.recipe(), H3Recipe::Turbo8);
    let mut record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let parameters = record["video_parameters"].as_array().unwrap();
    assert_eq!(parameters.len(), 703);
    for (i, p) in parameters.iter().enumerate() {
        assert_eq!(p["name"], format!("w{i}"));
        assert_eq!(p["dtype"], "float32");
        let bytes = p["shape"]
            .as_array()
            .unwrap()
            .iter()
            .try_fold(4u64, |n, d| n.checked_mul(d.as_u64().unwrap()))
            .unwrap();
        assert!(bytes > 0 && bytes <= corsairs::MAX_TENSOR_BYTES as u64);
        assert_eq!(
            std::fs::metadata(p["path"].as_str().unwrap())
                .unwrap()
                .len(),
            bytes
        );
    }
    record["video_parameters"].as_array_mut().unwrap().pop();
    let temporary = tempfile::tempdir().unwrap();
    let invalid = temporary.path().join("bundle.json");
    std::fs::write(&invalid, serde_json::to_vec(&record).unwrap()).unwrap();
    assert!(H3Assets::open(invalid).is_err());
}

#[test]
#[ignore = "requires AETHER_H3_SPATIAL_ORACLE; original source spatial stitch and patch extraction, no model inference"]
fn spatial_tiles_match_original_stitching_and_odd_latent_patch_offsets() {
    use aether::h3::{H3SpatialStitcher, H3SpatialTiling};
    use sha2::{Digest, Sha256};
    let path = std::path::PathBuf::from(
        std::env::var_os("AETHER_H3_SPATIAL_ORACLE").expect("set AETHER_H3_SPATIAL_ORACLE"),
    );
    let oracle: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let tiling: H3SpatialTiling = serde_json::from_value(oracle["tiling"].clone()).unwrap();
    let size = ImageSize::new(960, 544);
    let read = |item: &serde_json::Value| {
        let bytes = std::fs::read(item["path"].as_str().unwrap()).unwrap();
        let hash = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(hash, item["sha256"]);
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>()
    };
    let packed = read(&oracle["packed_rows"]);
    let mut stitcher = H3SpatialStitcher::new(tiling.clone(), size, 22).unwrap();
    for (i, tile) in oracle["tiles"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            tiling.extract(size, &packed, i).unwrap(),
            read(&tile["rows"])
        );
        stitcher.push(read(&tile["pixels"])).unwrap();
    }
    let output = stitcher.finish().unwrap();
    let hash = Sha256::digest(
        output
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect::<String>();
    assert_eq!(hash, oracle["reference_sha256"]);
    assert!(
        H3SpatialStitcher::new(tiling.clone(), size, 22)
            .unwrap()
            .finish()
            .is_err()
    );
    assert!(tiling.extract(size, &packed, tiling.tile_count()).is_err());
}

#[test]
fn bounded_media_writer_preserves_float_pcm_and_refuses_incomplete_clips() {
    let temp = tempfile::tempdir().unwrap();
    let mut writer =
        AudioVideoWriter::create(temp.path().join("clip"), ImageSize::new(8, 8), 3, 30).unwrap();
    for value in [0, 127, 255] {
        writer.write_frame(&[value; 8 * 8 * 3]).unwrap();
    }
    let samples: Vec<f32> = (0..3200)
        .flat_map(|i| [(i as f32 / 100.0).sin() * 1.1, -0.0])
        .collect();
    let audio = PcmAudio::new(32000, 2, samples.clone()).unwrap();
    let clip = writer.finish(&audio).unwrap();
    assert_eq!(
        (
            clip.frame_count(),
            clip.fps(),
            clip.sample_rate(),
            clip.channels()
        ),
        (3, 30, 32000, 2)
    );
    let wav = std::fs::read(clip.audio_path()).unwrap();
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(
        u32::from_le_bytes(wav[4..8].try_into().unwrap()) as usize + 8,
        wav.len()
    );
    assert_eq!(&wav[50..54], b"data");
    assert_eq!(
        &wav[58..],
        samples
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert!(AudioVideoWriter::create(clip.directory(), ImageSize::new(8, 8), 3, 30).is_err());
    let writer =
        AudioVideoWriter::create(temp.path().join("partial"), ImageSize::new(8, 8), 3, 30).unwrap();
    assert!(writer.finish(&audio).is_err());
    assert!(PcmAudio::new(0, 2, vec![0.0; 2]).is_err());
    assert!(PcmAudio::new(32000, 2, vec![f32::NAN; 2]).is_err());
    assert!(PcmAudio::new(32000, 2, vec![0.0; 3]).is_err());
}

#[test]
#[ignore = "requires AETHER_H3_LAB; reads existing lab fixtures without native execution"]
fn h3_saved_inputs_and_all_fifty_joint_schedule_pairs_are_verified() {
    let root = std::env::var_os("AETHER_H3_LAB").expect("set AETHER_H3_LAB");
    let lab = H3Lab::open(root).unwrap();
    let report = lab.inspect_inputs(128).unwrap();
    assert_eq!(report["missing_block_assets"], serde_json::json!([]));
    let steps: Vec<_> = lab.base_schedule().unwrap().steps().collect();
    assert_eq!(steps.len(), 50);
    assert_eq!(
        (steps[0].video_timestep, steps[0].audio_timestep),
        (0.0, 0.0)
    );
    assert_eq!(
        steps[1].video_timestep.to_bits(),
        0.001_697_838_3_f32.to_bits()
    );
    assert_eq!(
        steps[1].audio_timestep.to_bits(),
        0.006_756_723_f32.to_bits()
    );
    assert!(
        steps
            .iter()
            .all(|s| s.video_delta < 0.0 && s.audio_delta < 0.0)
    );
}

#[test]
#[ignore = "requires AETHER_H3_LAB; hashes retained backbone fixtures, no native execution"]
fn h3_backbone_inputs_verify_export_table_and_segment_identity() {
    let root =
        std::path::PathBuf::from(std::env::var_os("AETHER_H3_LAB").expect("set AETHER_H3_LAB"));
    let lab = H3Lab::open(&root).unwrap();
    let report = lab.inspect_backbone().unwrap();
    assert_eq!(report["blocks"], 50);
    assert_eq!(report["tables"], 50);
    assert_eq!(report["missing_segment_assets"], serde_json::json!([]));
    assert_eq!(report["model_streams_hashed"], false);
    assert_eq!(
        report["expected_output_sha256"],
        "e4aabaed96045e8b86cd292f7c6234f4ad5d86dda2e33edc41330d3e9772e75b"
    );

    // Copy only the small records. Mismatched provenance must fail before any
    // fixture/model access, rather than merely reporting a missing large asset.
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join("records")).unwrap();
    for name in [
        "environment.json",
        "block-oracle-19350.json",
        "backbone-export.json",
        "backbone-tables.json",
        "segments-25-export.json",
        "backbone-streaming.json",
    ] {
        std::fs::copy(
            root.join("records").join(name),
            temp.path().join("records").join(name),
        )
        .unwrap();
    }
    let copied = H3Lab::open(temp.path()).unwrap();
    let tables = temp.path().join("records/backbone-tables.json");
    let original = std::fs::read(&tables).unwrap();
    let mut changed: serde_json::Value = serde_json::from_slice(&original).unwrap();
    changed["tembSHA256"] = "0".repeat(64).into();
    std::fs::write(&tables, serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(
        copied
            .inspect_backbone()
            .unwrap_err()
            .to_string()
            .contains("different export or timestep pair")
    );
    std::fs::write(tables, original).unwrap();
    let segments = temp.path().join("records/segments-25-export.json");
    let mut changed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&segments).unwrap()).unwrap();
    changed["segments"].as_array_mut().unwrap().swap(0, 1);
    std::fs::write(segments, serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(
        copied
            .inspect_backbone()
            .unwrap_err()
            .to_string()
            .contains("segment order, range or asset path")
    );
}

#[test]
#[ignore = "requires AETHER_H3_LAB; compares every actual packed row and all timestep assignments"]
fn h3_layout_matches_all_four_captured_workflows_and_all_fifty_timesteps() {
    let root =
        std::path::PathBuf::from(std::env::var_os("AETHER_H3_LAB").expect("set AETHER_H3_LAB"));
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("records/packed-layout.json")).unwrap())
            .unwrap();
    let schedule = H3Schedule::from_lab(root.join("records/scheduler-50-forward.json")).unwrap();
    for case in record["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let anchors = match name {
            "first" => vec![H3Anchor::First],
            "last" => vec![H3Anchor::Last],
            "both" => vec![H3Anchor::First, H3Anchor::Last],
            _ => vec![],
        };
        let mut request = AudioVideoRequest::new("", ImageSize::new(960, 544), 124, 0);
        if anchors.contains(&H3Anchor::First) {
            request.first_frame = Some("first.png".into());
        }
        if anchors.contains(&H3Anchor::Last) {
            request.last_frame = Some("last.png".into());
        }
        let tag_item = case["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "text_token_tags")
            .unwrap();
        let tag_bytes = std::fs::read(root.join(tag_item["file"].as_str().unwrap())).unwrap();
        let tags: Vec<_> = tag_bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as i32)
            .collect();
        let layout = H3Layout::new(
            H3Geometry::for_request(&request, false).unwrap(),
            &tags,
            &anchors,
        )
        .unwrap();
        for (field, actual) in [
            ("token_tags", &layout.token_tags),
            ("video_indices", &layout.video_indices),
            ("audio_indices", &layout.audio_indices),
            ("text_indices", &layout.text_indices),
        ] {
            let item = case["tensors"]
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["name"] == field)
                .unwrap();
            let bytes = std::fs::read(root.join(item["file"].as_str().unwrap())).unwrap();
            let expected: Vec<_> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as i32)
                .collect();
            assert_eq!(*actual, expected, "{name} {field}");
        }
        let item = case["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "position_ids")
            .unwrap();
        let bytes = std::fs::read(root.join(item["file"].as_str().unwrap())).unwrap();
        let expected: Vec<_> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b) as f32)
            .collect();
        assert_eq!(layout.positions, expected, "{name} rotary coordinates");
        if name == "text" {
            for step in schedule.steps() {
                let plan = layout.timesteps(step);
                let field = format!("row_timestep_plan_{}_1", step.index);
                let item = case["tensors"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|t| t["name"] == field)
                    .unwrap();
                let bytes = std::fs::read(root.join(item["file"].as_str().unwrap())).unwrap();
                let expected: Vec<_> = bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_le_bytes(*b) as i32)
                    .collect();
                assert_eq!(plan.indices, expected, "step {}", step.index);
                let values: Vec<f32> = case["state"]["row_timestep_plan"][step.index][0]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect();
                assert_eq!(plan.values, values);
                assert_eq!(plan.values.len(), if step.index == 0 { 1 } else { 2 });
            }
        }
    }
}

#[test]
fn h3_scheduler_and_patch_order_match_independent_upstream_oracle() {
    use sha2::{Digest, Sha256};
    let record: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/h3-numerics.json")).unwrap();
    let floats = |value: &serde_json::Value| {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>()
    };
    let hash = |values: &[f32]| {
        Sha256::digest(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
    };
    let schedule = H3Schedule::from_json(include_bytes!("fixtures/h3-schedule.json")).unwrap();
    let mut video = floats(&record["video"]);
    let mut audio = floats(&record["audio"]);
    for (i, step) in record["steps"].as_array().unwrap().iter().enumerate() {
        schedule
            .advance_video(i, &mut video, &floats(&step["video_velocity"]))
            .unwrap();
        schedule
            .advance_audio(i, &mut audio, &floats(&step["audio_velocity"]))
            .unwrap();
        assert_eq!(
            hash(&video),
            step["video_sha256"].as_str().unwrap(),
            "video step {i}"
        );
        assert_eq!(
            hash(&audio),
            step["audio_sha256"].as_str().unwrap(),
            "audio step {i}"
        );
    }
    let request = AudioVideoRequest::new("", ImageSize::new(64, 64), 22, 0);
    let geometry = H3Geometry::for_request(&request, true).unwrap();
    let latent: Vec<_> = (0..24 * 7 * 4 * 4).map(|i| i as f32).collect();
    assert_eq!(
        hash(&pack_video(&latent, geometry).unwrap()),
        record["patchify_sha256"].as_str().unwrap()
    );
    let decode = &record["decode_stitching"];
    let mut stitcher = aether::h3::H3VideoStitcher::new(ImageSize::new(2, 2), 124).unwrap();
    let mut actual = Vec::new();
    for window in decode["windows"].as_array().unwrap() {
        actual.extend(stitcher.push(&floats(window)).unwrap());
    }
    actual.extend(stitcher.finish().unwrap());
    assert_eq!(actual.len(), 124 * 3 * 2 * 2);
    assert_eq!(hash(&actual), decode["reference_sha256"].as_str().unwrap());
    let mut invalid = aether::h3::H3VideoStitcher::new(ImageSize::new(2, 2), 22).unwrap();
    assert!(invalid.push(&[0.0; 4]).is_err());
    assert!(invalid.finish().is_err());
}

#[test]
#[ignore = "requires AETHER_FFMPEG and AETHER_FFPROBE; creates a small synthetic media clip, no model inference"]
fn mux_contains_both_video_and_audio_streams() {
    let ffmpeg = std::env::var_os("AETHER_FFMPEG").expect("set AETHER_FFMPEG");
    let ffprobe = std::env::var_os("AETHER_FFPROBE").expect("set AETHER_FFPROBE");
    let temp = tempfile::tempdir().unwrap();
    let mut writer =
        AudioVideoWriter::create(temp.path().join("clip"), ImageSize::new(16, 16), 3, 30).unwrap();
    for value in [10, 80, 160] {
        writer.write_frame(&vec![value; 16 * 16 * 3]).unwrap();
    }
    let audio = PcmAudio::new(32000, 2, vec![0.0; 6400]).unwrap();
    let clip = writer.finish(&audio).unwrap();
    let output = temp.path().join("clip.mp4");
    clip.mux_mp4(&ffmpeg, &output).unwrap();
    let probe = std::process::Command::new(ffprobe)
        .args(["-v", "error", "-show_streams", "-of", "json"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(probe.status.success());
    let result: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
    let streams = result["streams"].as_array().unwrap();
    assert!(streams.iter().any(|s| s["codec_type"] == "audio"));
    assert!(streams.iter().any(|s| s["codec_type"] == "video"
        && s["width"] == 16
        && s["height"] == 16
        && s["nb_frames"] == "3"));
    assert!(clip.mux_mp4(ffmpeg, output).is_err());
}
