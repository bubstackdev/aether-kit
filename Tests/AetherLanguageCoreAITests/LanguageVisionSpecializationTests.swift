@testable import AetherLanguageCoreAI
import Testing

struct LanguageVisionSpecializationTests {
    /// The vision encoder is never specialized with expectFrequentReshapes: it computes wrong outputs
    /// on the GPU with it (F-P3-26). Absent or false requests resolve to false; true is refused while
    /// no qualification has shown it correct.
    @Test func encoderIsNeverOpenedWithFrequentReshapes() throws {
        #expect(LanguageVisionSpecialization.reshapesQualified == false)
        #expect(try LanguageVisionSpecialization.expectFrequentReshapes(requested: nil) == false)
        #expect(try LanguageVisionSpecialization.expectFrequentReshapes(requested: false) == false)
        #expect(throws: LanguageGenerationError.self) { try LanguageVisionSpecialization.expectFrequentReshapes(requested: true) }
    }
}
