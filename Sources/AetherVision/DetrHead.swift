public import AetherCore
public import AetherRuntime
import CoreGraphics
import Foundation

/// `detector-detr@1` (MANIFEST.md §1.3, §2.8): DETR-style detection from one static graph (RF-DETR nano and seg-nano).
/// Loads an ``ObjectDetectorModel``; its ``ObjectDetectorModel/detector()`` is the ``ObjectDetector``.
public enum DetrDetectorDriver: ModelDriver {
    public static let id: DriverID = .detectorDetr

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> ObjectDetectorModel {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        _ = try ObjectDetector.Configuration(manifest)
        return ObjectDetectorModel(
            bundle: bundle, manifest: manifest, device: options.device,
            options: .init(cachePolicy: options.cachePolicy, cache: options.cache))
    }
}

/// The `detrSigmoid` postprocess (upstream RF-DETR's `PostProcess`, then `predict`'s threshold):
/// 1. **Scores:** the sigmoid of every `classLogits` value, `[Q, C]`.
/// 2. **Selection** (`topKPairs`): every (query, class) pair is a candidate; the `maxDetections` best by score
///    descending, ties by flat index `query · C + class` ascending, are kept when their score is strictly above
///    `scoreThreshold`. A query can yield several classes; there is no NMS. Columns with no label (COCO's unassigned
///    ids, which a COCO model never scores) are left out before ranking, so up to `maxDetections` labelled ones return.
/// 3. **Boxes:** the pair's query box, `cxcywh` or `xyxy`, `normalized` to the graph's input or in its pixels
///    (`letterboxPixels`), mapped to the source image through the prepared image's geometry and clamped to [0, 1].
/// 4. **Masks** (when the graph binds `masks`): the query's logit plane, which spans the graph's whole input, resampled
///    bilinearly onto the source image's pixels (pixel centres, `F.interpolate(align_corners=False)` for a stretched
///    input; clamped to the content's edge cells when letterboxed, so padding is never read) and kept strictly above
///    `masks.logitThreshold`. Only the plane's region that can exceed it is resampled (bilinear values never exceed
///    their four cells'), so the cost follows the object's size, not the image's.
struct DetrHead: Sendable {
    let postprocess: Manifest.Postprocess.DetrSigmoid
    let boxes: String, classLogits: String, masks: String?
    let labels: DetectionLabels

    /// Checks the manifest after ``DriverContract/detectorDetr`` has.
    init(_ manifest: Manifest, labels: DetectionLabels) throws(AetherError) {
        guard case .detrSigmoid(let postprocess)? = manifest.postprocess else {
            throw .manifestInvalid(field: "postprocess", reason: "\(manifest.driver) requires kind detrSigmoid")
        }
        let graph = manifest.graphs["detector"]!
        boxes = graph.outputs["boxes"]!.name
        classLogits = graph.outputs["classLogits"]!.name
        masks = graph.outputs["masks"]?.name
        switch (masks, postprocess.masks) {
        case (.some, nil):
            throw .manifestInvalid(field: "postprocess.masks", reason: "required when graphs.detector binds masks")
        case (nil, .some):
            throw .manifestInvalid(field: "postprocess.masks", reason: "graphs.detector binds no masks output")
        default:
            break
        }
        self.postprocess = postprocess
        self.labels = labels
    }

    var classIndex: Manifest.Postprocess.ClassIndex { postprocess.classIndex ?? .contiguous }

    /// The detector graph's outputs to bind, by graph name.
    var outputs: [String] { [boxes, classLogits] + (masks.map { [$0] } ?? []) }

    /// Throws unless the loaded graph's outputs are `[1, Q, 4]`, `[1, Q, C]` and `[1, Q, h, w]`, with C the label set's
    /// size for contiguous columns and no more columns than its largest id for `cocoId`.
    func check(_ signature: GraphSignature) throws(AetherError) {
        func shape(_ name: String) -> [Int]? { signature.output(name)?.shape }
        func mismatch(_ expected: String, _ name: String) -> AetherError {
            .manifestMismatch(field: "graphs.detector.outputs", expected: "'\(name)' of shape \(expected)",
                              found: shape(name).map { "\($0)" } ?? "no such output")
        }
        guard let b = shape(boxes), b.count == 3, b[0] == 1, b[1] > 0, b[2] == 4 else { throw mismatch("[1, Q, 4]", boxes) }
        let columns = classIndex == .contiguous ? labels.labels.count : (labels.labels.map(\.id).max() ?? -1) + 1
        guard let l = shape(classLogits), l.count == 3, l[0] == 1, l[1] == b[1], l[2] > 0,
              classIndex == .contiguous ? l[2] == columns : l[2] <= columns else {
            throw mismatch("[1, \(b[1]), C] with C \(classIndex == .contiguous ? "=" : "≤") \(columns) (the label set's)", classLogits)
        }
        if let masks {
            guard let m = shape(masks), m.count == 4, m[0] == 1, m[1] == b[1], m[2] > 0, m[3] > 0 else {
                throw mismatch("[1, \(b[1]), h, w]", masks)
            }
        }
    }

    /// The detections, best first.
    func detections(
        _ outputs: [String: DetectorOutput], geometry: ImageGeometry, scoreThreshold: Double, maxDetections: Int
    ) -> [Detection] {
        let logits = outputs[classLogits]!, box = outputs[boxes]!.values
        let columns = logits.shape[2]
        let named = (0..<columns).map { labels.label(column: $0, classIndex: classIndex) }
        let picked = Self.select(logits.values, columns: columns, labelled: named.map { $0 != nil },
                                 threshold: Float(scoreThreshold), limit: maxDetections)
        let mapper = DetectionBoxes(format: postprocess.boxFormat, space: postprocess.boxSpace, geometry: geometry)
        let plane = masks.flatMap { outputs[$0] }
        let resampler = plane.map { MaskResampler(geometry, planeShape: $0.shape) }
        let maskThreshold = Float(postprocess.masks?.logitThreshold ?? 0)
        var planes: [Int: DetectionMask] = [:]
        return picked.map { pick in
            let query = pick.index / columns
            var mask: DetectionMask?
            if let resampler, let plane {
                mask = planes[query] ?? resampler.mask(plane.values, plane: query, threshold: maskThreshold)
                planes[query] = mask
            }
            let v = (0..<4).map { Double(box[4 * query + $0]) }
            return Detection(label: named[pick.index % columns]!, score: pick.score,
                             box: mapper.source(v[0], v[1], v[2], v[3]), mask: mask)
        }
    }

    /// The `topKPairs` rule over the labelled columns: the flat indices scoring strictly above `threshold`, the best
    /// `limit` of them, by score descending then index ascending. The same rows as ranking every pair and then
    /// thresholding: every pair kept outscores every pair dropped.
    static func select(
        _ logits: [Float], columns: Int, labelled: [Bool], threshold: Float, limit: Int
    ) -> [(index: Int, score: Float)] {
        var candidates: [(index: Int, score: Float)] = []
        for (i, logit) in logits.enumerated() where labelled[i % columns] {
            let score = 1 / (1 + exp(-logit))
            if score > threshold { candidates.append((i, score)) }
        }
        candidates.sort { $0.score != $1.score ? $0.score > $1.score : $0.index < $1.index }
        return Array(candidates.prefix(max(0, limit)))
    }
}

/// The graph's raw output, as the head reads it.
struct DetectorOutput: Sendable {
    var shape: [Int]
    var values: [Float]
}

/// Mask planes (`[1, Q, h, w]`, spanning the graph's whole input) onto the source image's pixels: bilinear at pixel
/// centres through the prepared image's geometry, with ``DepthResampler``'s taps scaled to the plane.
struct MaskResampler: Sendable {
    let planeWidth: Int, planeHeight: Int
    let sourceWidth: Int, sourceHeight: Int
    let columns: [(Int, Int, Float)], rows: [(Int, Int, Float)]

    init(_ geometry: ImageGeometry, planeShape: [Int]) {
        planeHeight = planeShape[2]
        planeWidth = planeShape[3]
        sourceWidth = geometry.sourceWidth
        sourceHeight = geometry.sourceHeight
        let sx = Double(planeWidth) / Double(geometry.width), sy = Double(planeHeight) / Double(geometry.height)
        columns = DepthResampler.taps(count: sourceWidth, origin: geometry.content.minX * sx,
                                      extent: geometry.content.width * sx, limit: planeWidth)
        rows = DepthResampler.taps(count: sourceHeight, origin: geometry.content.minY * sy,
                                   extent: geometry.content.height * sy, limit: planeHeight)
    }

    /// The source pixels whose taps touch `low...high` (taps are monotonic).
    static func span(_ taps: [(Int, Int, Float)], _ low: Int, _ high: Int) -> Range<Int> {
        guard let first = taps.firstIndex(where: { $0.1 >= low && $0.0 <= high }) else { return 0..<0 }
        let last = taps.lastIndex(where: { $0.1 >= low && $0.0 <= high })!
        return first..<(last + 1)
    }

    /// Plane `index` of `planes`, foreground strictly above `threshold`.
    func mask(_ planes: [Float], plane index: Int, threshold: Float) -> DetectionMask {
        let size = planeWidth * planeHeight, base = index * size
        // The cells above the threshold bound where an interpolated value can be.
        var c0 = Int.max, c1 = -1, r0 = Int.max, r1 = -1
        for r in 0..<planeHeight {
            for c in 0..<planeWidth where planes[base + r * planeWidth + c] > threshold {
                c0 = min(c0, c)
                c1 = max(c1, c)
                r0 = min(r0, r)
                r1 = max(r1, r)
            }
        }
        let empty = DetectionMask(width: sourceWidth, height: sourceHeight, bounds: .init(x: 0, y: 0, width: 0, height: 0), bits: [])
        guard c1 >= 0 else { return empty }
        let xs = Self.span(columns, c0, c1), ys = Self.span(rows, r0, r1)
        var hits: [(Int, Int)] = []
        var left = Int.max, right = -1, top = Int.max, bottom = -1
        planes.withUnsafeBufferPointer { m in
            for y in ys {
                let (a, b, fy) = rows[y]
                let upperRow = base + a * planeWidth, lowerRow = base + b * planeWidth
                for x in xs {
                    let (p, q, fx) = columns[x]
                    let upper = m[upperRow + p] + (m[upperRow + q] - m[upperRow + p]) * fx
                    let lower = m[lowerRow + p] + (m[lowerRow + q] - m[lowerRow + p]) * fx
                    if upper + (lower - upper) * fy > threshold {
                        hits.append((x, y))
                        left = min(left, x)
                        right = max(right, x)
                        top = min(top, y)
                        bottom = max(bottom, y)
                    }
                }
            }
        }
        guard right >= 0 else { return empty }
        let width = right - left + 1, height = bottom - top + 1
        var bits = [UInt8](repeating: 0, count: (width * height + 7) / 8)
        for (x, y) in hits {
            let i = (y - top) * width + (x - left)
            bits[i >> 3] |= 1 << UInt8(i & 7)
        }
        return DetectionMask(width: sourceWidth, height: sourceHeight,
                             bounds: .init(x: left, y: top, width: width, height: height), bits: bits)
    }
}
