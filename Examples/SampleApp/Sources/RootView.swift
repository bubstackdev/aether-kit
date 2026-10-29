import SampleAppCore
import SwiftUI

struct RootView: View {
    let app: SampleAppState

    var body: some View {
        TabView {
            Tab("Model", systemImage: "shippingbox") { NavigationStack { ModelScreen(model: app.model) } }
            Tab("Chat", systemImage: "bubble.left.and.bubble.right") { NavigationStack { ChatScreen(chat: app.chat) } }
            Tab("Structured", systemImage: "curlybraces") { NavigationStack { StructuredScreen(todo: app.todo) } }
        }
        .tabViewStyle(.sidebarAdaptable)
        .safeAreaInset(edge: .bottom) { LifecycleStrip(lifecycle: app.model.lifecycle) }
    }
}

/// The model's lifecycle and the process's memory pressure, live, on every screen.
struct LifecycleStrip: View {
    let lifecycle: LifecycleDisplay

    var body: some View {
        HStack(spacing: 12) {
            Label(lifecycle.model, systemImage: lifecycle.isLoaded ? "memorychip.fill" : "memorychip")
            Spacer()
            Label(lifecycle.pressure, systemImage: "gauge.with.dots.needle.33percent")
        }
        .font(.footnote)
        .foregroundStyle(.secondary)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.bar)
        .accessibilityElement(children: .combine)
    }
}

struct ModelScreen: View {
    let model: ModelController

    var body: some View {
        let display = model.display
        Form {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Label(display.title, systemImage: icon(display.tone))
                        .font(.headline)
                        .foregroundStyle(color(display.tone))
                    Text(display.detail).foregroundStyle(.secondary)
                    if let progress = display.progress {
                        ProgressView(value: progress)
                    } else if display.tone == .busy {
                        ProgressView()
                    }
                }
                .padding(.vertical, 4)
            } header: {
                Text(model.entry.displayName ?? "Model")
            } footer: {
                Text("Qwen3-0.6B, converted by AetherKit from Qwen/Qwen3-0.6B (Apache-2.0), downloaded from Hugging Face and verified file by file.")
            }
            if !display.actions.isEmpty {
                Section {
                    ForEach(display.actions, id: \.self) { action in
                        Button(title(action), role: action == .delete ? .destructive : nil) {
                            Task { await model.perform(action) }
                        }
                    }
                }
            }
            if let error = model.actionError {
                Section { Text(error).foregroundStyle(.red) }
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Model")
    }

    private func title(_ action: ReadinessDisplay.Action) -> String {
        switch action {
        case .download: model.display.title == "Paused" ? "Resume download" : "Download"
        case .cancel: "Cancel"
        case .retry: "Retry"
        case .delete: "Delete model"
        case .refresh: "Check again"
        }
    }

    private func icon(_ tone: ReadinessDisplay.Tone) -> String {
        switch tone {
        case .neutral: "arrow.down.circle"
        case .busy: "hourglass"
        case .ready: "checkmark.seal"
        case .problem: "exclamationmark.triangle"
        }
    }

    private func color(_ tone: ReadinessDisplay.Tone) -> Color {
        switch tone {
        case .neutral, .busy: .primary
        case .ready: .green
        case .problem: .orange
        }
    }
}

struct ChatScreen: View {
    let chat: ChatController
    @State private var draft = ""

    var body: some View {
        VStack(spacing: 0) {
            if !chat.isAvailable {
                ContentUnavailableView("No model yet", systemImage: "shippingbox", description: Text("Download the model on the Model screen."))
            } else {
                ScrollViewReader { proxy in
                    List(chat.messages) { message in
                        MessageRow(message: message).id(message.id)
                    }
                    .listStyle(.plain)
                    .onChange(of: chat.messages.last?.text) { proxy.scrollTo(chat.messages.last?.id, anchor: .bottom) }
                }
                HStack {
                    TextField("Message", text: $draft, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...4)
                        .onSubmit(send)
                    if chat.isResponding {
                        Button("Stop", systemImage: "stop.circle.fill") { chat.stop() }
                    } else {
                        Button("Send", systemImage: "arrow.up.circle.fill", action: send)
                            .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
                .labelStyle(.iconOnly)
                .font(.title2)
                .padding()
            }
        }
        .navigationTitle("Chat")
        .toolbar {
            Button("New chat", systemImage: "square.and.pencil") { Task { await chat.newChat() } }
                .disabled(!chat.isAvailable || chat.messages.isEmpty)
        }
    }

    private func send() {
        chat.send(draft)
        draft = ""
    }
}

struct MessageRow: View {
    let message: ChatMessage

    var body: some View {
        VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
            Text(message.text.isEmpty && message.isStreaming ? "…" : message.text)
                .textSelection(.enabled)
                .padding(10)
                .background(message.role == .user ? Color.accentColor.opacity(0.15) : Color.secondary.opacity(0.1),
                            in: RoundedRectangle(cornerRadius: 12))
            if let footnote = message.footnote {
                Text(footnote).font(.caption2).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
        .listRowSeparator(.hidden)
    }
}

struct StructuredScreen: View {
    let todo: TodoController
    @State private var sentence = TodoController.example

    var body: some View {
        Form {
            Section {
                TextField("Sentence", text: $sentence, axis: .vertical).lineLimit(2...5)
                Button("Extract a to-do item") { todo.run(sentence) }
                    .disabled(!todo.isAvailable || todo.isWorking || sentence.isEmpty)
            } header: {
                Text("Sentence")
            } footer: {
                Text("Generation is constrained to TodoItem's JSON schema, so the reply always decodes.")
            }
            if todo.isWorking {
                Section { ProgressView() }
            }
            if let item = todo.item {
                Section("TodoItem") {
                    LabeledContent("Title", value: item.title)
                    LabeledContent("Priority", value: item.priority.rawValue)
                    LabeledContent("Due", value: item.dueInDays.map { "in \($0) days" } ?? "not given")
                    if let seconds = todo.seconds { LabeledContent("Took", value: String(format: "%.2f s", seconds)) }
                }
                if let json = todo.json {
                    Section("JSON") { Text(json).font(.caption.monospaced()).textSelection(.enabled) }
                }
            }
            if let error = todo.error {
                Section { Text(error).foregroundStyle(.red) }
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Structured")
        .overlay {
            if !todo.isAvailable {
                ContentUnavailableView("No model yet", systemImage: "shippingbox", description: Text("Download the model on the Model screen."))
            }
        }
    }
}
