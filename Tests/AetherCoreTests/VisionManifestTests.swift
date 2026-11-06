import Foundation
import Testing
@testable import AetherCore

/// MANIFEST.md §2.5 (`preprocess.image`) and §2.8 (`postprocess`, `labels`): typed, closed vocabularies, and errors
/// that name their field.
@Suite("Vision manifest blocks")
struct VisionManifestTests {
    /// A manifest with the given top-level blocks (JSON object members).
    static func manifest(_ blocks: String) -> Data {
        Data("""
        { "schema": 1, "id": "v", "task": "t", "driver": "depth-estimator@1", "variant": { "platform": "macos" },
          "assets": { "main": { "subtree": "" } },
          "graphs": { "estimator": { "file": "main:g.aimodel", "function": "main", "inputs": { "image": "image" },
                                     "outputs": { "depth": "depth" }, "compute": "gpu" } },
          \(blocks) }
        """.utf8)
    }

    /// DA3's Forge export writes these blocks (`x-` fields and all).
    @Test func theDepthExportersBlocksDecode() throws {
        let m = try Manifest.decode(Self.manifest("""
        "preprocess": { "image": { "resize": { "mode": "stretch", "size": [504, 504], "filter": "vImage" }, "colorSpace": "sRGB",
                                   "channelOrder": "RGB", "scale": 0.00392156862745098, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
                                   "layout": "NCHW", "hostDType": "float32", "orientation": "exif", "x-notes": "n" } },
        "postprocess": { "kind": "depth", "depth": { "polarity": "largerIsFarther", "resizeBack": true }, "x-notes": "n" }
        """))
        let image = try #require(m.preprocess?.image)
        #expect(image.resize == .init(mode: .stretch, size: .init(width: 504, height: 504), filter: .vImage))
        #expect(image.colorSpace == .sRGB && image.channelOrder == .RGB && image.layout == .NCHW && image.hostDType == .float32)
        #expect(image.scale == 1.0 / 255 && image.mean == [0, 0, 0] && image.std == [1, 1, 1] && image.orientation == .exif)
        #expect(m.postprocess == .depth(.init(polarity: .largerIsFarther, resizeBack: true)))
        #expect(try Manifest.decode(m.encoded()) == m)
    }

    @Test func everyPostprocessKindDecodesAndRoundTrips() throws {
        let cases: [(String, Manifest.Postprocess)] = [
            (#"{ "kind": "depth", "depth": { "polarity": "largerIsNearer" } }"#, .depth(.init(polarity: .largerIsNearer))),
            (#"{ "kind": "denseNms", "boxFormat": "cxcywh", "boxSpace": "letterboxPixels", "nms": { "type": "perClass", "iou": 0.45 }, "defaults": { "scoreThreshold": 0.3, "maxDetections": 50 } }"#,
             .denseNms(.init(boxFormat: .cxcywh, boxSpace: .letterboxPixels, nms: .init(type: .perClass, iou: 0.45),
                             defaults: .init(scoreThreshold: 0.3, maxDetections: 50)))),
            (#"{ "kind": "detrSigmoid", "boxFormat": "cxcywh", "boxSpace": "normalized", "classIndex": "cocoId", "selection": "topKPairs", "defaults": { "scoreThreshold": 0.5, "maxDetections": 50 }, "masks": {} }"#,
             .detrSigmoid(.init(boxFormat: .cxcywh, boxSpace: .normalized, defaults: .init(scoreThreshold: 0.5, maxDetections: 50),
                                classIndex: .cocoId, selection: .topKPairs, masks: .init()))),
            (#"{ "kind": "embedding", "embedding": { "normalized": "inGraph", "similarity": "cosine" } }"#,
             .embedding(.init(normalized: .inGraph, similarity: .cosine))),
            (#"{ "kind": "gliner" }"#, .gliner(.init())),
            (#"{ "kind": "yesNo", "probs": { "positiveIndex": 1, "threshold": 0.5 } }"#, .yesNo(.init(positiveIndex: 1, threshold: 0.5))),
            (#"{ "kind": "maxSim", "retrieval": { "imageTokenSpan": [13, 77], "patchGrid": [8, 8] } }"#,
             .maxSim(.init(imageTokenSpan: [13, 77], patchGrid: [8, 8]))),
            (#"{ "kind": "superRes", "sr": { "range": [-1, 1], "colorMatch": "host" } }"#, .superRes(.init(range: [-1, 1], colorMatch: .host))),
        ]
        #expect(Set(cases.map(\.1.kind)) == Set(Manifest.Postprocess.Kind.allCases))
        for (json, expected) in cases {
            let m = try Manifest.decode(Self.manifest(#""postprocess": "# + json))
            #expect(m.postprocess == expected, "\(json)")
            #expect(try Manifest.decode(m.encoded()) == m, "\(json)")
        }
    }

    @Test func invalidPostprocessAndLabelsNameTheirField() {
        func expect(_ field: String, _ json: String, sourceLocation: SourceLocation = #_sourceLocation) {
            expectField(field, sourceLocation: sourceLocation) { _ = try Manifest.decode(Self.manifest(json)) }
        }
        expect("postprocess.kind", #""postprocess": { "kind": "nms" }"#)
        expect("postprocess.depth", #""postprocess": { "kind": "depth" }"#)
        expect("postprocess.depth.polarity", #""postprocess": { "kind": "depth", "depth": { "polarity": "up" } }"#)
        expect("postprocess.nms.iou", #""postprocess": { "kind": "denseNms", "boxFormat": "xyxy", "boxSpace": "normalized", "nms": { "type": "perClass", "iou": 0 }, "defaults": { "scoreThreshold": 0.3, "maxDetections": 5 } }"#)
        expect("postprocess.defaults.maxDetections", #""postprocess": { "kind": "detrSigmoid", "boxFormat": "xyxy", "boxSpace": "normalized", "defaults": { "scoreThreshold": 0.3, "maxDetections": 0 } }"#)
        expect("postprocess.selection", #""postprocess": { "kind": "detrSigmoid", "boxFormat": "xyxy", "boxSpace": "normalized", "selection": "perQuery", "defaults": { "scoreThreshold": 0.3, "maxDetections": 5 } }"#)
        expect("postprocess.probs.threshold", #""postprocess": { "kind": "yesNo", "probs": { "positiveIndex": 1, "threshold": 2 } }"#)
        expect("postprocess.retrieval.patchGrid", #""postprocess": { "kind": "maxSim", "retrieval": { "imageTokenSpan": [13, 77], "patchGrid": [8, 7] } }"#)
        expect("postprocess.sr.range", #""postprocess": { "kind": "superRes", "sr": { "range": [1, -1] } }"#)
        expect("labels", #""labels": { "set": "coco-91", "file": "main:labels.json" }"#)
        expect("labels.set", #""labels": { "set": "imagenet" }"#)
        expect("labels.file", #""labels": { "file": "other:labels.json" }"#)
        expect("preprocess.image.resize.size", #""preprocess": { "image": { "resize": { "mode": "letterbox", "filter": "vImage" } } }"#)
        expect("preprocess.image.resize.filter", #""preprocess": { "image": { "resize": { "mode": "stretch", "size": [2, 2], "filter": "lanczos" } } }"#)
        expect("preprocess.image.resize.size", #""preprocess": { "image": { "resize": { "mode": "stretch", "size": [2], "filter": "vImage" } } }"#)
    }

    @Test func labelsAreABuiltInSetOrAFile() throws {
        #expect(try Manifest.decode(Self.manifest(#""labels": { "set": "coco-91" }"#)).labels == .set(.coco91))
        let file = try Manifest.decode(Self.manifest(#""labels": { "file": "main:labels.json" }"#))
        #expect(file.labels == .file(AssetPath(asset: "main", path: "labels.json")))
        #expect(try Manifest.decode(file.encoded()) == file)
    }
}
