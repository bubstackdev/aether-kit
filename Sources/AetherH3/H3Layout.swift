// Derived from the Apache-2.0 Aether prototype. See NOTICE.
import AetherKit
import Foundation

/// Keyframe row placement only; this module does not encode keyframe images.
public enum H3Anchor: Sendable, Equatable { case first, last }

/// Packed text, optional condition video, stereo audio and target video row plan.
/// Coordinates use source float64 arithmetic and narrow once to float32 for rotary input.
public struct H3Layout: Sendable {
    public let positions: FloatTensor
    public let tokenTags: [Int32]
    public let videoIndices: [Int32]
    public let audioIndices: [Int32]
    public let textIndices: [Int32]
    public let conditionVideoRows: Int
    public let audioLatentsPerChannel: Int
    public var rows: Int { tokenTags.count }

    public struct RowTimesteps: Sendable {
        public let values: [Float]
        public let indices: [Int32]
        public let adalnIndices: [Int32]
    }

    public init(geometry: H3Geometry, textTags: [Int32], anchors: [H3Anchor] = []) throws {
        guard (1...4096).contains(textTags.count), textTags.allSatisfy({ $0 == 0 || $0 == 1 }),
              anchors.count <= 2, anchors.count < 2 || anchors == [.first, .last] else {
            throw AetherError.invalidRequest("H3 requires 1...4096 text tags (0/1), with keyframes ordered first then last")
        }
        let n = textTags.count, h = geometry.latentHeight, w = geometry.latentWidth
        let frameRows = (h / 2) * (w / 2)
        conditionVideoRows = frameRows * anchors.count
        audioLatentsPerChannel = geometry.audioLatentsPerChannel
        let audioStart = n + conditionVideoRows
        let videoStart = audioStart + audioLatentsPerChannel * 2
        let rows = videoStart + geometry.targetVideoRows
        var positions = Array(repeating: Float(0), count: rows * 3)
        for i in 0..<n { positions[i * 3] = Float(i) }
        let sqrtArea = Double(h * w).squareRoot()
        func grid(_ dimension: Int) -> [Double] {
            let ratio = Double(dimension) / sqrtArea
            let left = (1 - ratio) / 2
            let step = ((left + ratio) - left) / Double(dimension / 2)
            return (0..<(dimension / 2)).map { (left + Double($0) * step) * 32 }
        }
        let hg = grid(h), wg = grid(w)
        let spans = (0..<geometry.latentFrames).map { (5.0 / 3.0) * ($0 % 5 == 0 ? 1.0 : 4.0) }
        func frame(start: Int, time: Double) {
            for (y, hy) in hg.enumerated() {
                for (x, wx) in wg.enumerated() {
                    let at = (start + y * wg.count + x) * 3
                    positions[at] = Float(time); positions[at + 1] = Float(hy); positions[at + 2] = Float(wx)
                }
            }
        }
        for (i, anchor) in anchors.enumerated() {
            let time = anchor == .first ? Double(n) : Double(n) + spans.reduce(0, +) - 5.0 / 3.0
            frame(start: n + i * frameRows, time: time)
        }
        var cumulative = 0.0
        for (i, span) in spans.enumerated() {
            frame(start: videoStart + i * frameRows, time: Double(n) + cumulative)
            cumulative += span
        }
        for channel in 0..<2 {
            for t in 0..<audioLatentsPerChannel {
                let at = (audioStart + channel * audioLatentsPerChannel + t) * 3
                positions[at] = Float(n + t)
                positions[at + 2] = Float(channel == 0 ? wg[0] : wg[wg.count - 1])
            }
        }
        var tags = textTags + Array(repeating: Int32(0), count: rows - n)
        for i in audioStart..<videoStart { tags[i] = 2 }
        self.positions = try FloatTensor(shape: [rows, 3], values: positions)
        tokenTags = tags
        textIndices = (0..<n).map(Int32.init)
        audioIndices = (audioStart..<videoStart).map(Int32.init)
        videoIndices = (n..<audioStart).map(Int32.init) + (videoStart..<rows).map(Int32.init)
    }

    /// Exact timestep cardinality/order and per-row modulation selection. Keyframe
    /// rows use max(video time, 0.999); text inherits video time.
    public func timesteps(for step: H3Schedule.Step) -> RowTimesteps {
        var perRow = Array(repeating: step.videoTimestep, count: rows)
        for i in audioIndices { perRow[Int(i)] = step.audioTimestep }
        for i in videoIndices.prefix(conditionVideoRows) { perRow[Int(i)] = max(step.videoTimestep, 0.999) }
        let values = Array(Set(perRow)).sorted()
        let indices = perRow.map { value in Int32(values.firstIndex(of: value)!) }
        return RowTimesteps(values: values, indices: indices,
            adalnIndices: zip(indices, tokenTags).map { $0 * 3 + $1 })
    }
}

extension H3Layout {
    /// The transformer's 3-axis rotary tables `[rows, 96]` (Diffusers `MiniMaxH3RotaryPosEmbed`, 16 frequencies per
    /// axis at θ = 10000, the t/h/w angle blocks concatenated, then repeated for `rotate_half`). float32 throughout.
    public func rotary() throws -> (cos: FloatTensor, sin: FloatTensor) {
        let inverse = (0..<16).map { Float(1) / powf(10000, Float(2 * $0) / 32) }
        var cos = [Float](repeating: 0, count: rows * 96), sin = cos
        for row in 0..<rows {
            for axis in 0..<3 {
                let position = positions.values[row * 3 + axis]
                for f in 0..<16 {
                    let angle = position * inverse[f]
                    let (c, s) = (cosf(angle), sinf(angle))
                    for copy in 0..<2 { cos[row * 96 + copy * 48 + axis * 16 + f] = c; sin[row * 96 + copy * 48 + axis * 16 + f] = s }
                }
            }
        }
        return (try FloatTensor(shape: [rows, 96], values: cos), try FloatTensor(shape: [rows, 96], values: sin))
    }
}
