public import Foundation

/// Audio a generative driver produced (Kokoro's speech, H3's soundtrack): planar float32 PCM, kept unclipped as the
/// model produced it. Mono and multichannel producers share the layout.
public struct GeneratedAudio: Sendable, Hashable {
    /// WAV sample encodings. `pcm16` clips to `[-1, 1]` and rounds ties to even; `float32` stores the unclipped
    /// samples (`WAVE_FORMAT_IEEE_FLOAT`).
    public enum WAVEncoding: Sendable, Hashable { case pcm16, float32 }

    public let sampleRate: Int
    /// One array of samples per channel, all the same length.
    public let channels: [[Float]]
    public var channelCount: Int { channels.count }
    public var frameCount: Int { channels[0].count }
    public var duration: Duration { .seconds(Double(frameCount) / Double(sampleRate)) }
    /// The first channel; for mono audio, every sample.
    public var samples: [Float] { channels[0] }

    /// Fails unless there are 1–8 equal, non-empty channels of finite samples at a rate in 1...768,000 Hz.
    public init(channels: [[Float]], sampleRate: Int) throws(AetherError) {
        guard (1...768_000).contains(sampleRate), (1...8).contains(channels.count), let first = channels.first,
              !first.isEmpty, channels.allSatisfy({ $0.count == first.count }) else {
            throw .tensor(name: "audio", reason: .unsupported(
                "audio needs 1-8 equal, non-empty channels and a sample rate in 1...768000"))
        }
        guard channels.allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
            throw .tensor(name: "audio", reason: .unsupported("the audio has a non-finite sample"))
        }
        self.channels = channels
        self.sampleRate = sampleRate
    }

    public init(mono samples: [Float], sampleRate: Int) throws(AetherError) {
        try self.init(channels: [samples], sampleRate: sampleRate)
    }

    /// Joins audio with identical layout in order (consecutive synthesized chunks, say).
    public static func concatenating(_ parts: [GeneratedAudio]) throws(AetherError) -> GeneratedAudio {
        guard let first = parts.first,
              parts.allSatisfy({ $0.sampleRate == first.sampleRate && $0.channelCount == first.channelCount }) else {
            throw .tensor(name: "audio", reason: .unsupported("audio parts must share their sample rate and channel count"))
        }
        let channels = (0..<first.channelCount).map { channel in parts.flatMap { $0.channels[channel] } }
        return try GeneratedAudio(channels: channels, sampleRate: first.sampleRate)
    }

    /// RIFF/WAVE bytes with interleaved frames. Mono and stereo use `WAVE_FORMAT_PCM` (a 16-byte `fmt `) or
    /// `WAVE_FORMAT_IEEE_FLOAT` (an 18-byte `fmt ` with `cbSize` 0, plus the `fact` chunk non-PCM formats require).
    /// More than two channels use `WAVE_FORMAT_EXTENSIBLE` with an unspecified speaker mask, plus `fact` for float.
    public func wavData(encoding: WAVEncoding = .pcm16) throws(AetherError) -> Data {
        let bytesPerSample = encoding == .pcm16 ? 2 : 4
        let extensible = channelCount > 2
        let float = encoding == .float32
        let formatBytes = extensible ? 40 : (float ? 18 : 16)
        let factBytes = (float || extensible) ? 12 : 0
        let header = 12 + 8 + formatBytes + factBytes + 8
        let (payload, overflow) = frameCount.multipliedReportingOverflow(by: channelCount * bytesPerSample)
        guard !overflow, payload <= Int(UInt32.max) - header else {
            throw .capacity(what: "WAV file bytes", needed: Int64(clamping: payload), available: Int64(UInt32.max) - Int64(header))
        }
        var data = Data(count: header + payload)
        data.withUnsafeMutableBytes { raw in
            var offset = 0
            func bytes(_ text: String) { for byte in text.utf8 { raw[offset] = byte; offset += 1 } }
            func u16(_ value: Int) { raw.storeBytes(of: UInt16(value).littleEndian, toByteOffset: offset, as: UInt16.self); offset += 2 }
            func u32(_ value: Int) { raw.storeBytes(of: UInt32(value).littleEndian, toByteOffset: offset, as: UInt32.self); offset += 4 }
            bytes("RIFF"); u32(header - 8 + payload); bytes("WAVE")
            bytes("fmt "); u32(formatBytes)
            u16(extensible ? 0xFFFE : (float ? 3 : 1)); u16(channelCount); u32(sampleRate)
            u32(sampleRate * channelCount * bytesPerSample); u16(channelCount * bytesPerSample); u16(bytesPerSample * 8)
            if extensible {
                u16(22); u16(bytesPerSample * 8); u32(0)
                // KSDATAFORMAT_SUBTYPE_PCM / _IEEE_FLOAT: {0000000X-0000-0010-8000-00AA00389B71}
                u32(float ? 3 : 1); u16(0); u16(0x10)
                for byte: UInt8 in [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71] { raw[offset] = byte; offset += 1 }
            } else if float {
                u16(0)
            }
            if factBytes > 0 { bytes("fact"); u32(4); u32(frameCount) }
            bytes("data"); u32(payload)
            let stride = channelCount * bytesPerSample
            for channel in 0..<channelCount {
                var position = offset + channel * bytesPerSample
                for value in channels[channel] {
                    switch encoding {
                    case .pcm16:
                        let scaled = Int16((min(max(value, -1), 1) * 32767).rounded(.toNearestOrEven))
                        raw.storeBytes(of: scaled.littleEndian, toByteOffset: position, as: Int16.self)
                    case .float32:
                        raw.storeBytes(of: value.bitPattern.littleEndian, toByteOffset: position, as: UInt32.self)
                    }
                    position += stride
                }
            }
        }
        return data
    }
}
