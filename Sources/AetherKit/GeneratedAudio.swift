import Foundation

/// Finite planar float32 PCM, retained unclipped as the model produced it.
///
/// This is the kit's audio delivery value: speech synthesis returns mono, and multichannel
/// producers can use the same planar layout. Samples use Swift value ownership.
public struct GeneratedAudio: Sendable, Equatable {
    /// WAV sample encodings. `pcm16` clips to [-1, 1] and rounds ties to even; `float32`
    /// stores the unclipped samples (WAVE_FORMAT_IEEE_FLOAT).
    public enum WAVEncoding: Sendable { case pcm16, float32 }

    public let sampleRate: Int
    public let channels: [[Float]]
    public var channelCount: Int { channels.count }
    public var frameCount: Int { channels[0].count }
    public var durationSeconds: Double { Double(frameCount) / Double(sampleRate) }
    /// The first channel; for mono audio, every sample.
    public var samples: [Float] { channels[0] }

    public init(channels: [[Float]], sampleRate: Int) throws {
        guard (1...768_000).contains(sampleRate), (1...8).contains(channels.count), let first = channels.first,
              !first.isEmpty, channels.allSatisfy({ $0.count == first.count }) else {
            throw AetherError.invalidTensor("audio needs 1-8 equal, non-empty channels and a valid sample rate")
        }
        guard channels.allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
            throw AetherError.invalidTensor("audio samples must be finite")
        }
        self.channels = channels
        self.sampleRate = sampleRate
    }

    public init(mono samples: [Float], sampleRate: Int) throws {
        try self.init(channels: [samples], sampleRate: sampleRate)
    }

    /// Joins audio with identical layout in order, for example consecutive synthesized chunks.
    public static func concatenating(_ parts: [GeneratedAudio]) throws -> GeneratedAudio {
        guard let first = parts.first,
              parts.allSatisfy({ $0.sampleRate == first.sampleRate && $0.channelCount == first.channelCount }) else {
            throw AetherError.invalidRequest("audio parts must share sample rate and channel count")
        }
        let channels = (0..<first.channelCount).map { channel in parts.flatMap { $0.channels[channel] } }
        return try GeneratedAudio(channels: channels, sampleRate: first.sampleRate)
    }

    /// RIFF/WAVE bytes with interleaved frames.
    public func wavData(encoding: WAVEncoding = .pcm16) throws -> Data {
        let bytesPerSample = encoding == .pcm16 ? 2 : 4
        let (payload, overflow) = frameCount.multipliedReportingOverflow(by: channelCount * bytesPerSample)
        guard !overflow, payload <= Int(UInt32.max) - 36 else {
            throw AetherError.invalidRequest("audio is too long for a WAV file")
        }
        var data = Data(capacity: 44 + payload)
        func u32(_ value: Int) { withUnsafeBytes(of: UInt32(value).littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ value: Int) { withUnsafeBytes(of: UInt16(value).littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8)); u32(36 + payload)
        data.append(contentsOf: Array("WAVEfmt ".utf8)); u32(16)
        u16(encoding == .pcm16 ? 1 : 3); u16(channelCount); u32(sampleRate)
        u32(sampleRate * channelCount * bytesPerSample); u16(channelCount * bytesPerSample); u16(bytesPerSample * 8)
        data.append(contentsOf: Array("data".utf8)); u32(payload)
        var interleaved = [UInt8]()
        interleaved.reserveCapacity(payload)
        for frame in 0..<frameCount {
            for channel in 0..<channelCount {
                let value = channels[channel][frame]
                switch encoding {
                case .pcm16:
                    let scaled = (min(max(value, -1), 1) * 32767).rounded(.toNearestOrEven)
                    withUnsafeBytes(of: Int16(scaled).littleEndian) { interleaved.append(contentsOf: $0) }
                case .float32:
                    withUnsafeBytes(of: value.bitPattern.littleEndian) { interleaved.append(contentsOf: $0) }
                }
            }
        }
        data.append(contentsOf: interleaved)
        return data
    }

    /// Writes atomically; the destination's directory must exist.
    public func saveWAV(to url: URL, encoding: WAVEncoding = .pcm16) throws {
        guard url.isFileURL else { throw AetherError.invalidRequest("WAV destination must be local") }
        try wavData(encoding: encoding).write(to: url, options: .atomic)
    }
}
