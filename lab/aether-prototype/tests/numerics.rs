use aether::{
    GeneratedImage, ImageSize,
    krea::{FlowSchedule, KreaPrompt, seeded_noise},
};

#[test]
fn schedule_matches_independent_swift_bits() {
    let schedule = FlowSchedule::turbo(8).unwrap();
    // KreaScheduleReference.swift / Corsairs acceptance oracle, not Rust-derived.
    assert_eq!(
        schedule
            .timesteps
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [
            1065353216, 1064627160, 1063751508, 1062674713, 1061318479, 1059557808, 1057180097,
            1050619501
        ]
    );
    assert_eq!(
        schedule
            .sigmas
            .windows(2)
            .map(|s| (s[1] - s[0]).to_bits())
            .collect::<Vec<_>>(),
        [
            3174122112, 3176515648, 3179508184, 3181743696, 3184979192, 3188793276, 3192834602,
            3198103149
        ]
    );
    assert_eq!(FlowSchedule::turbo(1).unwrap().sigmas, [1.0, 0.0]);
    assert!(FlowSchedule::turbo(0).is_err());
    assert!(FlowSchedule::turbo(51).is_err());
}

#[test]
fn euler_uses_separately_rounded_product() {
    let schedule = FlowSchedule {
        sigmas: vec![1.0, 0.0],
        timesteps: vec![1.0],
    };
    // (1 + ulp) * (1 - ulp) rounds to 1 before adding -1.
    let a = f32::from_bits(0x3f800001);
    let b = f32::from_bits(0x3f7ffffe);
    let custom = FlowSchedule {
        sigmas: vec![0.0, a],
        ..schedule.clone()
    };
    let mut x = [-1.0];
    custom.advance(0, &mut x, &[b]).unwrap();
    assert_eq!(x[0], 0.0);
    assert_ne!(a.mul_add(b, -1.0), 0.0);
    assert!(schedule.advance(1, &mut x, &[0.0]).is_err());
    assert!(schedule.advance(0, &mut x, &[f32::NAN]).is_err());
}

#[test]
fn noise_is_prefix_stable_and_finite() {
    let short = seeded_noise(71344, 3);
    let long = seeded_noise(71344, 8192);
    assert_eq!(short, long[..3]);
    assert_ne!(short, seeded_noise(71345, 3));
    assert!(long.iter().all(|v| v.is_finite()));
    assert!(seeded_noise(0, 0).is_empty());
    let variance = long.iter().map(|v| v * v).sum::<f32>() / long.len() as f32;
    assert!((variance - 1.0).abs() < 0.05);
}

#[test]
fn noise_matches_swift_seeded_noise_bits() {
    // Recorded by running the lab's unchanged SeededNoise.swift on this host.
    let cases = [
        (
            0,
            [
                3202863069, 1045741727, 1076470663, 3204126899, 3212645673, 1072668933, 1048658635,
                3219995906,
            ],
        ),
        (
            42,
            [
                1054103078, 1059526174, 3211023016, 1068094895, 1071473486, 3220247501, 1057729992,
                3218346802,
            ],
        ),
        (
            71344,
            [
                1035851364, 1074166398, 1060776933, 3184718423, 1070719263, 3206437512, 1025533962,
                1052832527,
            ],
        ),
    ];
    for (seed, expected) in cases {
        assert_eq!(
            seeded_noise(seed, 8)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn image_rounding_matches_swift_and_png_is_readable() {
    let image =
        GeneratedImage::from_chw(ImageSize::new(2, 1), vec![-1.0, 1.0, 0.0, 2.0, -2.0, 0.0])
            .unwrap();
    assert_eq!(image.rgb8(), [0, 128, 0, 255, 255, 128]);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("image.png");
    image.save_png(&path).unwrap();
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut bytes = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut bytes).unwrap();
    assert_eq!((info.width, info.height), (2, 1));
    assert_eq!(&bytes[..info.buffer_size()], image.rgb8());
    assert!(GeneratedImage::from_chw(ImageSize::new(1, 1), vec![f32::NAN; 3]).is_err());
}

#[test]
fn prompt_id_shape_limits() {
    assert!(KreaPrompt::from_token_ids(vec![0; 34]).is_err());
    assert!(KreaPrompt::from_token_ids(vec![0; 547]).is_err());
    assert!(KreaPrompt::from_token_ids(vec![-1; 35]).is_err());
    assert_eq!(
        KreaPrompt::from_token_ids(vec![0; 54]).unwrap().text_rows(),
        20
    );
}
