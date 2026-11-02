import AetherAssets
import AetherKit
import AetherLanguage
import Foundation
import Observation

/// The app's state: the model screen, and the chat and structured screens that use its model. When the model changes,
/// the screens let go of the old one before the model screen goes on (so a delete isn't refused as busy).
@MainActor
@Observable
public final class SampleAppState {
    public let model: ModelController
    public let chat = ChatController()
    public let todo = TodoController()

    public init(model: ModelController) {
        self.model = model
        model.modelDidChange = { [weak self] newModel in
            guard let self else { return }
            await self.chat.attach(model: newModel)
            await self.todo.attach(model: newModel)
        }
    }

    /// The shipped entry for this platform, in the app's store, with the platform's transport.
    public static func live(lifecycle: ModelLifecyclePolicy = .platformDefault) throws -> SampleAppState {
        let preparation = ModelPreparation(store: try ModelController.defaultStore(), transport: ModelController.defaultTransport())
        return SampleAppState(model: ModelController(entry: try SampleCatalog.entry(), preparation: preparation, lifecycle: lifecycle))
    }
}
