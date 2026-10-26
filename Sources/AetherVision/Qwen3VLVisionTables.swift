/// Host-side per-patch tables for the Qwen3-VL vision encoder (`encode`'s `pos_index`,
/// `pos_weight` and `patch_rc` inputs). Pure and Core-AI-independent.
///
/// Every row is in merge-window order: merge x merge windows in raster order, the patches inside a
/// window in raster order, which is the order of the preprocessor's pixel rows. A still image has
/// one temporal group; with several, the tables repeat per group.
///
/// - `positionIndex` / `positionWeight` [n, 4]: bilinear sampling of the encoder's learned
///   side x side position table at y = linspace(0, side - 1, h), x = linspace(0, side - 1, w)
///   (corners aligned), neighbours ordered (y0x0, y0x1, y1x0, y1x1). The encoder gathers the rows.
/// - `patchCoordinates` [n, 2]: each patch's integer (row, column), driving its 2-D rotary.
public struct Qwen3VLVisionTables: Sendable, Equatable {
    public let grid: (t: Int, h: Int, w: Int)
    public let positionIndex: [Int32]
    public let positionWeight: [Float]
    public let patchCoordinates: [Int32]

    public var patchCount: Int { grid.t * grid.h * grid.w }

    public static func == (a: Self, b: Self) -> Bool {
        a.grid == b.grid && a.positionIndex == b.positionIndex && a.positionWeight == b.positionWeight
            && a.patchCoordinates == b.patchCoordinates
    }

    public enum Failure: Error, Equatable { case invalidGrid(t: Int, h: Int, w: Int, merge: Int) }

    /// Tables for a patch grid of `t` temporal groups by `h` x `w` patches (both multiples of `merge`).
    public init(t: Int, h: Int, w: Int, merge: Int = 2, tableSide: Int = 48) throws {
        guard t > 0, h > 0, w > 0, merge > 0, h % merge == 0, w % merge == 0, tableSide > 1 else {
            throw Failure.invalidGrid(t: t, h: h, w: w, merge: merge)
        }
        let ys = Self.axis(h, side: tableSide), xs = Self.axis(w, side: tableSide)
        var index: [Int32] = [], weight: [Float] = [], coordinates: [Int32] = []
        index.reserveCapacity(t * h * w * 4); weight.reserveCapacity(t * h * w * 4); coordinates.reserveCapacity(t * h * w * 2)
        for _ in 0..<t {
            for windowRow in 0..<(h / merge) { for windowColumn in 0..<(w / merge) {
                for i in 0..<merge { for j in 0..<merge {
                    let row = windowRow * merge + i, column = windowColumn * merge + j
                    let (y0, y1, dy) = ys[row], (x0, x1, dx) = xs[column]
                    index += [y0 * tableSide + x0, y0 * tableSide + x1, y1 * tableSide + x0, y1 * tableSide + x1].map { Int32($0) }
                    weight += [(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx].map { Float($0) }
                    coordinates += [Int32(row), Int32(column)]
                } }
            } }
        }
        self.grid = (t, h, w)
        self.positionIndex = index
        self.positionWeight = weight
        self.patchCoordinates = coordinates
    }

    /// (lower neighbour, upper neighbour, fraction) per position of `linspace(0, side - 1, n)`,
    /// computed like numpy: `i * step` in double precision with the last sample exactly `side - 1`.
    private static func axis(_ n: Int, side: Int) -> [(Int, Int, Double)] {
        let stop = Double(side - 1)
        return (0..<n).map { i in
            let position = n == 1 ? 0 : (i == n - 1 ? stop : Double(i) * (stop / Double(n - 1)))
            let lower = Int(position.rounded(.down))
            return (lower, min(lower + 1, side - 1), position - Double(lower))
        }
    }
}
