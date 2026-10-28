// usage: swift foreground_mask.swift <in-image> <out.png>
// Writes the image as RGBA with Vision's foreground-instance mask as alpha: the background removal the kit would use
// instead of upstream's rembg (u2net). An image that already has transparency is copied through unchanged, as
// upstream's remove_background does.
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

let arguments = CommandLine.arguments
guard arguments.count == 3 else { print("usage: swift foreground_mask.swift <in-image> <out.png>"); exit(2) }
let input = URL(fileURLWithPath: arguments[1]), output = URL(fileURLWithPath: arguments[2])
guard let source = CGImageSourceCreateWithURL(input as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { print("cannot read \(input.path)"); exit(1) }

func write(_ cgImage: CGImage) {
    guard let destination = CGImageDestinationCreateWithURL(output as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { print("cannot write \(output.path)"); exit(1) }
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else { print("cannot finalize \(output.path)"); exit(1) }
}

func hasTransparency(_ image: CGImage) -> Bool {
    switch image.alphaInfo {
    case .none, .noneSkipFirst, .noneSkipLast: return false
    default: break
    }
    let context = CIContext()
    let extent = CIImage(cgImage: image).extent
    guard let filter = CIFilter(name: "CIAreaMinimum", parameters: [kCIInputImageKey: CIImage(cgImage: image),
                                                                    kCIInputExtentKey: CIVector(cgRect: extent)]),
          let minimum = filter.outputImage else { return false }
    var pixel = [UInt8](repeating: 255, count: 4)
    context.render(minimum, toBitmap: &pixel, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                   format: .RGBA8, colorSpace: nil)
    return pixel[3] < 255
}

if hasTransparency(image) {
    write(image)
    print("kept the existing alpha: \(image.width)x\(image.height)")
    exit(0)
}
let handler = VNImageRequestHandler(cgImage: image)
let request = VNGenerateForegroundInstanceMaskRequest()
do {
    try handler.perform([request])
    guard let observation = request.results?.first else { print("no foreground found"); exit(1) }
    let masked = try observation.generateMaskedImage(ofInstances: observation.allInstances, from: handler,
                                                     croppedToInstancesExtent: false)
    let context = CIContext()
    guard let cgImage = context.createCGImage(CIImage(cvPixelBuffer: masked), from: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    else { print("cannot render the masked image"); exit(1) }
    write(cgImage)
    print("masked \(observation.allInstances.count) instance(s): \(image.width)x\(image.height)")
} catch {
    print("Vision failed: \(error)")
    exit(1)
}
