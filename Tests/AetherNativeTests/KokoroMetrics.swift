import Accelerate
import Foundation

/// Mirrors Forge/kokoro/speech_metrics.py (float64 throughout).
enum KokoroMetrics {
    static let sampleRate = 24_000.0
    static let nFFT = 1024
    static let hop = 256
    static let mels = 80
    static let fMax = 12_000.0
    static let floor = 1e-5

    struct Comparison: Codable, Equatable {
        let rel_l2: Double
        let max_abs: Double
        let snr_db: Double
        let log_mel_mean_abs: Double
        let log_mel_max_abs: Double
    }

    static func compare(_ x: [Float], _ r: [Float]) -> Comparison {
        precondition(x.count == r.count && x.count >= nFFT)
        var errNorm = 0.0, refNorm = 0.0, maxAbs = 0.0
        for i in 0..<x.count {
            let e = Double(x[i]) - Double(r[i])
            errNorm += e * e
            refNorm += Double(r[i]) * Double(r[i])
            maxAbs = max(maxAbs, abs(e))
        }
        let rel = (errNorm / refNorm).squareRoot()
        let a = logMel(x), b = logMel(r)
        var sum = 0.0, peak = 0.0
        for i in 0..<a.count {
            let d = abs(a[i] - b[i])
            sum += d
            peak = max(peak, d)
        }
        return Comparison(rel_l2: rel, max_abs: maxAbs, snr_db: rel > 0 ? 20 * log10(1 / rel) : .infinity,
                          log_mel_mean_abs: sum / Double(a.count), log_mel_max_abs: peak)
    }

    private static func hzToMel(_ hz: Double) -> Double { 2595 * log10(1 + hz / 700) }

    static let filterbank: [Double] = {  // [mels x bins]
        let bins = nFFT / 2 + 1
        let points = (0..<(mels + 2)).map { hzToMel(0) + (hzToMel(fMax) - hzToMel(0)) * Double($0) / Double(mels + 1) }
        let hz = points.map { 700 * (pow(10, $0 / 2595) - 1) }
        var bank = [Double](repeating: 0, count: mels * bins)
        for m in 0..<mels {
            for k in 0..<bins {
                let f = Double(k) * sampleRate / Double(nFFT)
                let up = (f - hz[m]) / (hz[m + 1] - hz[m])
                let down = (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1])
                bank[m * bins + k] = max(0, min(up, down))
            }
        }
        return bank
    }()

    static let dft: (cos: [Double], sin: [Double]) = {  // [nFFT x bins]
        let bins = nFFT / 2 + 1
        var c = [Double](repeating: 0, count: nFFT * bins), s = c
        for n in 0..<nFFT {
            for k in 0..<bins {
                let angle = 2 * Double.pi * Double((n * k) % nFFT) / Double(nFFT)
                c[n * bins + k] = cos(angle)
                s[n * bins + k] = sin(angle)
            }
        }
        return (c, s)
    }()

    static func logMel(_ x: [Float]) -> [Double] {
        let bins = nFFT / 2 + 1
        let frames = 1 + (x.count - nFFT) / hop
        let window = (0..<nFFT).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(nFFT)) }
        var stack = [Double](repeating: 0, count: frames * nFFT)
        for f in 0..<frames {
            for n in 0..<nFFT { stack[f * nFFT + n] = Double(x[f * hop + n]) * window[n] }
        }
        var re = [Double](repeating: 0, count: frames * bins), im = re
        vDSP_mmulD(stack, 1, dft.cos, 1, &re, 1, vDSP_Length(frames), vDSP_Length(bins), vDSP_Length(nFFT))
        vDSP_mmulD(stack, 1, dft.sin, 1, &im, 1, vDSP_Length(frames), vDSP_Length(bins), vDSP_Length(nFFT))
        var magnitude = [Double](repeating: 0, count: frames * bins)
        for i in 0..<magnitude.count { magnitude[i] = (re[i] * re[i] + im[i] * im[i]).squareRoot() }
        // mel[f, m] = sum_k magnitude[f, k] * bank[m, k]
        var transposed = [Double](repeating: 0, count: bins * mels)
        vDSP_mtransD(filterbank, 1, &transposed, 1, vDSP_Length(bins), vDSP_Length(mels))
        var mel = [Double](repeating: 0, count: frames * mels)
        vDSP_mmulD(magnitude, 1, transposed, 1, &mel, 1, vDSP_Length(frames), vDSP_Length(mels), vDSP_Length(bins))
        return mel.map { log(max($0, floor)) }
    }
}
