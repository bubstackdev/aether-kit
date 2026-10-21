//! Optional read-only checks against local external inputs; no Core AI needed.
use aether::{
    ImageSize,
    krea::{KreaAssets, KreaTokenizer, image_rotary},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn env_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name}"))
}

#[test]
#[ignore = "requires AETHER_KREA_BUNDLE and AETHER_KREA_FIXTURES"]
fn every_saved_prompt_matches_token_ids_and_truncation_preserves_suffix() {
    let bundle = env_path("AETHER_KREA_BUNDLE");
    KreaAssets::open(&bundle).unwrap();
    let tokenizer = KreaTokenizer::open(bundle.join("tokenizer/tokenizer.json")).unwrap();
    let fixtures = env_path("AETHER_KREA_FIXTURES");
    let cases = std::fs::read_to_string(fixtures.join("cases.jsonl")).unwrap();
    for line in cases.lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        let prompt = tokenizer.prepare(case["prompt"].as_str().unwrap()).unwrap();
        let bytes = std::fs::read(fixtures.join(case["files"]["ids"].as_str().unwrap())).unwrap();
        let ids: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        assert_eq!(prompt.token_ids(), ids, "{}", case["id"]);
        assert_eq!(
            prompt.text_rows(),
            case["textShape"][0].as_u64().unwrap() as usize
        );
        assert!(!prompt.truncated());
    }
    let short = tokenizer.prepare("hello").unwrap();
    let long = tokenizer.prepare(&"hello ".repeat(2000)).unwrap();
    assert!(long.truncated());
    assert_eq!(long.token_ids().len(), 546);
    assert_eq!(
        &long.token_ids()[541..],
        &short.token_ids()[short.token_ids().len() - 5..]
    );
}

#[test]
#[ignore = "requires AETHER_KREA_LAB"]
fn rotary_matches_every_saved_swift_byte() {
    let reference = std::fs::read(
        env_path("AETHER_KREA_LAB").join("records/round4/block-inputs-tower/rope_image.f32"),
    )
    .unwrap();
    let values = image_rotary(ImageSize::new(768, 1344)).unwrap();
    let bytes: Vec<_> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    assert_eq!(Sha256::digest(&bytes), Sha256::digest(&reference));
    assert_eq!(bytes, reference);
}
