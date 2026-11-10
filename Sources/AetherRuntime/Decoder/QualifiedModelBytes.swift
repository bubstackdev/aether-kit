import AetherCore

extension DecoderEngine {
    /// What a qualified model-memory figure was measured on: the export (its ``Manifest/contentDigest``) and the
    /// Core AI architecture of the chip. The digest covers the variant's platform and every graph file's hash, so
    /// another export (a requantization, another recipe, the other platform's variant) or another chip doesn't use it.
    public struct QualifiedExport: Sendable, Hashable {
        public var exportDigest: String
        public var architecture: String

        public init(exportDigest: String, architecture: String) {
            self.exportDigest = exportDigest
            self.architecture = architecture
        }
    }

    /// The kit's measured model memory for known `decoder-lm@1` and `vlm-static-inject@1` exports, keyed by the
    /// export and the chip it was measured on (as ``Memory/measuredModelBytes`` after a gate's session). A load from a
    /// bundle whose manifest and device match an entry, and whose ``Options/modelBytes`` is nil, declares the entry's
    /// figure, so the charge is the model's measured memory from the first load, without a host having to keep the
    /// measurement and pass it back. An explicit ``Options/modelBytes`` takes precedence (0 charges what load measures,
    /// as without an entry).
    ///
    /// Each entry comes from a native gate, which prints it ready to add (`GATE qualified model bytes entry:`): the
    /// describer's retention gate (`DescriberRetentionGateTests`, docs/VLM.md G5) for the `vlm-static-inject@1`
    /// export, and the Qwen3 export gate (`DecoderLMExportGateTests`) for the `decoder-lm@1` one. The gates check that
    /// an entry for their export still covers what they measure.
    public static let qualifiedModelBytes: [QualifiedExport: Int64] = [:]

    /// The qualified model bytes for `manifest` on `device` (``qualifiedModelBytes``), nil when the kit has none.
    public static func qualifiedModelBytes(for manifest: Manifest, device: Device) -> Int64? {
        qualifiedModelBytes(for: manifest, device: device, table: qualifiedModelBytes)
    }

    static func qualifiedModelBytes(for manifest: Manifest, device: Device, table: [QualifiedExport: Int64]) -> Int64? {
        guard manifest.driver == .decoderLM || manifest.driver == .vlmStaticInject else { return nil }
        return table[QualifiedExport(exportDigest: manifest.contentDigest, architecture: device.architecture)]
    }

    /// `options` with ``Options/modelBytes`` set from `table` when the caller declared none (a bundle load).
    static func declaringQualified(
        _ options: Options, manifest: Manifest, device: Device, table: [QualifiedExport: Int64] = qualifiedModelBytes
    ) -> Options {
        guard options.modelBytes == nil, let qualified = qualifiedModelBytes(for: manifest, device: device, table: table)
        else { return options }
        var options = options
        options.modelBytes = qualified
        return options
    }

    /// The entry a gate prints for ``qualifiedModelBytes``: `measured` rounded up to a whole MiB, written as the
    /// table's literal, for `manifest` on `device`.
    package static func qualifiedEntry(measured: Int64, manifest: Manifest, device: Device, note: String) -> String {
        let mib = (max(0, measured) + (1 << 20) - 1) >> 20
        return "QualifiedExport(exportDigest: \"\(manifest.contentDigest)\", architecture: \"\(device.architecture)\"): "
            + "\(mib) << 20, // \(manifest.id), \(note)"
    }
}
