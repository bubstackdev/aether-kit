/// Options every driver load receives.
public struct LoadOptions: Sendable, Hashable {
    public var device: Device
    public var cachePolicy: GraphOptions.CachePolicy
    public var cache: GraphOptions.Cache

    public init(device: Device, cachePolicy: GraphOptions.CachePolicy, cache: GraphOptions.Cache) {
        self.device = device
        self.cachePolicy = cachePolicy
        self.cache = cache
    }
}

/// A model family's loader, keyed by a versioned ``DriverID`` (DESIGN.md §6.3).
public protocol ModelDriver: Sendable {
    /// `decoder-lm@1`, `image-detector@1`, …
    static var id: DriverID { get }
    associatedtype Loaded: Sendable
    static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> Loaded
}

/// The set of drivers a build can load (DESIGN.md §1.3, §6.3).
///
/// Open registration without global state: a registry is an immutable value. Each task module exposes the drivers
/// it links, and the host composes them (`Registry([...]).merging(…)`), so adding a family never edits a central
/// `switch`. An unknown driver is ``AetherError/unsupported(driver:)``, never a fallback.
public struct Registry: Sendable {
    private var drivers: [DriverID: any ModelDriver.Type]

    /// A registry of `drivers`. When two share an id, the later one wins.
    public init(_ drivers: [any ModelDriver.Type] = []) {
        var map: [DriverID: any ModelDriver.Type] = [:]
        for driver in drivers { map[driver.id] = driver }
        self.drivers = map
    }

    /// No drivers: the runtime layer registers none itself. Task modules provide theirs.
    public static let empty = Registry()

    public var ids: [DriverID] { drivers.keys.sorted { $0.description < $1.description } }

    public func supports(_ id: DriverID) -> Bool { drivers[id] != nil }

    /// This registry plus `driver` (replacing one with the same id).
    public func registering(_ driver: any ModelDriver.Type) -> Registry {
        var copy = self
        copy.drivers[driver.id] = driver
        return copy
    }

    /// This registry plus every driver of `other` (`other` wins on a shared id).
    public func merging(_ other: Registry) -> Registry {
        var copy = self
        copy.drivers.merge(other.drivers) { _, theirs in theirs }
        return copy
    }

    /// The registered driver for `id`. A driver id this build knows but has no driver for is `.unsupported`; an id
    /// outside the driver catalog (MANIFEST.md §1) is `.unknownDriver`.
    public func driver(for id: DriverID) throws(AetherError) -> any ModelDriver.Type {
        if let driver = drivers[id] { return driver }
        try DriverCatalog.requireKnown(id)
        throw .unsupported(driver: id)
    }

    /// Fails unless a driver for the resolved variant is registered. Call it after `ModelRef.resolve` and before
    /// installing, so an unsupported model never downloads.
    public func require(_ ref: ModelRef) throws(AetherError) {
        _ = try driver(for: ref.driver)
    }

    /// Loads `bundle` with the driver its manifest names.
    public func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> any Sendable {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        return try await Self.open(try driver(for: manifest.driver), bundle, options)
    }

    /// Loads `bundle` with `driver`, which must be registered and must be the one the manifest names.
    public func load<D: ModelDriver>(_ bundle: InstalledBundle, as driver: D.Type, options: LoadOptions) async throws -> D.Loaded {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        guard manifest.driver == D.id else {
            throw AetherError.manifestMismatch(field: "driver", expected: D.id.description, found: manifest.driver.description)
        }
        _ = try self.driver(for: D.id)
        return try await D.load(bundle, options: options)
    }

    private static func open<D: ModelDriver>(_ driver: D.Type, _ bundle: InstalledBundle, _ options: LoadOptions) async throws -> any Sendable {
        try await driver.load(bundle, options: options)
    }
}
