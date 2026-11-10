import AetherCore

extension DecoderEngine {
    /// What a qualified model-memory figure was measured on: the export (its ``Manifest/contentDigest``) and the
    /// Core AI architecture of the chip. The digest covers the variant's platform and every graph file's hash, so
    /// another export (a requantization, another recipe, the other platform's variant) or another chip doesn't use it.
    package struct QualifiedExport: Sendable, Hashable {
        package var exportDigest: String
        package var architecture: String

        package init(exportDigest: String, architecture: String) {
            self.exportDigest = exportDigest
            self.architecture = architecture
        }
    }

    /// The kit's measured model memory for known `decoder-lm@1` and `vlm-static-inject@1` exports, keyed by the
    /// export and the chip it was measured on (as ``Memory/measuredModelBytes`` after a gate's session). A load from a
    /// bundle whose manifest and device match an entry, with ``Options/ModelCharge/qualified`` (the default), declares
    /// the entry's figure, so the charge is the model's measured memory from the first load, without a host having to
    /// keep the measurement and pass it back. ``Options/ModelCharge/measured`` or ``Options/ModelCharge/declared(bytes:)``
    /// takes precedence. Package-only: hosts read an entry through ``qualifiedModelBytes(for:device:)``.
    ///
    /// Each entry comes from a native gate, which prints it ready to add (`GATE qualified model bytes entry:`): the
    /// describer's retention gate (`DescriberRetentionGateTests`, docs/VLM.md G5) for the `vlm-static-inject@1`
    /// export, and the Qwen3 export gate (`DecoderLMExportGateTests`) for the `decoder-lm@1` one. The gates check that
    /// an entry for their export still covers what they measure.
    package static let qualifiedModelBytesTable: [QualifiedExport: Int64] = [:]

    /// The kit's qualified model bytes for `manifest` on `device`: the model memory a native gate measured for this
    /// export (its ``Manifest/contentDigest``) on this chip's Core AI architecture, which a bundle load charges by
    /// default (``Options/ModelCharge/qualified``). Nil when the kit has none (another export or chip, or a driver
    /// other than `decoder-lm@1` and `vlm-static-inject@1`).
    public static func qualifiedModelBytes(for manifest: Manifest, device: Device) -> Int64? {
        qualifiedModelBytes(for: manifest, device: device, table: qualifiedModelBytesTable)
    }

    static func qualifiedModelBytes(for manifest: Manifest, device: Device, table: [QualifiedExport: Int64]) -> Int64? {
        guard manifest.driver == .decoderLM || manifest.driver == .vlmStaticInject else { return nil }
        return table[QualifiedExport(exportDigest: manifest.contentDigest, architecture: device.architecture)]
    }

    /// `options` with ``Options/ModelCharge/qualified`` resolved from `table` (a bundle load): the entry declared, or
    /// what load measures when there is none.
    static func declaringQualified(
        _ options: Options, manifest: Manifest, device: Device, table: [QualifiedExport: Int64] = qualifiedModelBytesTable
    ) -> Options {
        guard options.modelCharge == .qualified else { return options }
        var options = options
        options.modelCharge = qualifiedModelBytes(for: manifest, device: device, table: table).map { .declared(bytes: $0) } ?? .measured
        return options
    }

    /// The entry a gate prints for ``qualifiedModelBytesTable``: `measured` rounded up to a whole MiB, written as the
    /// table's literal, for `manifest` on `device`.
    package static func qualifiedEntry(measured: Int64, manifest: Manifest, device: Device, note: String) -> String {
        let mib = (max(0, measured) + (1 << 20) - 1) >> 20
        return "QualifiedExport(exportDigest: \"\(manifest.contentDigest)\", architecture: \"\(device.architecture)\"): "
            + "\(mib) << 20, // \(manifest.id), \(note)"
    }
}
