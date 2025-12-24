#!/usr/bin/env python3
import sys
import json
import requests
import base64
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QComboBox, QLabel, QFileDialog,
    QListWidget, QListWidgetItem, QMenu, QInputDialog,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QScrollArea, QMessageBox
)

from database.postgres import PostgresDB


# ================= THREADS =================
class DirectOllamaThread(QThread):
    token = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, model, messages):
        super().__init__()
        self.model = model
        self.messages = messages
        self.running = True
        self.last_response = ""

    def stop(self):
        self.running = False

    def run(self):
        try:
            url = "http://localhost:11434/api/chat"
            payload = {
                "model": self.model,
                "messages": self.messages,
                "stream": True,
                "options": {"temperature": 0.7}
            }

            response = ""
            with requests.post(url, json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not self.running:
                        break
                    if line:
                        data = json.loads(line.decode())
                        if "message" in data and "content" in data["message"]:
                            token = data["message"]["content"]
                            response += token
                            self.token.emit(token)
                        if data.get("done"):
                            break
            self.last_response = response.strip()
            self.finished.emit(self.last_response)
        except Exception as e:
            self.error.emit(str(e))


class CustomCrewThread(QThread):
    token = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, user_prompt, crew_config, history_messages):
        super().__init__()
        self.user_prompt = user_prompt
        self.crew_config = crew_config
        self.history_messages = history_messages
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        try:
            output = "# BOFFIN CUSTOM CREW REPORT\n\n"
            output += f"**User Request:** {self.user_prompt}\n\n---\n\n"
            context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.history_messages])
            previous = f"{context}\nUSER: {self.user_prompt}"

            for i, agent in enumerate(self.crew_config, 1):
                if not self.running:
                    break

                role = agent['role']
                model = agent['model']
                system_prompt = agent.get('system_prompt', '').strip()
                input_prompt = agent['input_prompt'].format(previous=previous)

                self.token.emit(f"\n[👤 {i}. {role} ({model}) Working...]\n")

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": input_prompt})

                agent_thread = DirectOllamaThread(model, messages)
                agent_thread.token.connect(self.token)
                agent_thread.error.connect(self.error)

                agent_thread.start()
                agent_thread.wait()  # পরবর্তী এজেন্টের জন্য output দরকার → অপেক্ষা করতেই হবে

                if not self.running:
                    break

                out = agent_thread.last_response.strip()
                if not out:
                    out = "[No response from model]"
                output += f"### {role} Output\n{out}\n\n---\n\n"
                previous = out

            if self.running:
                self.finished.emit(output)
            else:
                self.finished.emit("")

        except Exception as e:
            self.error.emit(str(e))


# ================= CREW CONFIG DIALOG =================
class CrewConfigDialog(QDialog):
    def __init__(self, models, config=None, crew_name="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Crew" if config else "Create New Crew")
        self.setMinimumSize(950, 650)
        self.models = models
        self.agent_widgets = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Crew Name:</b>"))
        self.name_edit = QLineEdit(crew_name)
        layout.addWidget(self.name_edit)

        scroll = QScrollArea()
        scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)

        add_btn = QPushButton("➕ Add Agent")
        add_btn.clicked.connect(self.add_agent)

        layout.addWidget(QLabel("<b>Agents (in order):</b>"))
        layout.addWidget(add_btn)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if config:
            for agent in config:
                self.add_agent(agent)
        else:
            self.add_agent()

    def add_agent(self, preset=None):
        frame = QWidget()
        frame.setStyleSheet("background:#222; border:1px solid #555; border-radius:10px; padding:12px; margin:8px;")
        form = QFormLayout(frame)

        role = QLineEdit(); role.setPlaceholderText("e.g. Planner, Coder")
        model = QComboBox(); model.addItems(self.models)
        system = QTextEdit(); system.setFixedHeight(80)
        inp = QTextEdit(); inp.setFixedHeight(130)
        inp.setPlaceholderText("Use {previous} for previous output")

        if preset:
            role.setText(preset.get('role', ''))
            model.setCurrentText(preset.get('model', ''))
            system.setPlainText(preset.get('system_prompt', ''))
            inp.setPlainText(preset.get('input_prompt', ''))

        form.addRow("👤 Role:", role)
        form.addRow("🤖 Model:", model)
        form.addRow("📝 System:", system)
        form.addRow("📋 Input Template:", inp)

        self.agent_widgets.append({'role': role, 'model': model, 'system': system, 'input': inp, 'frame': frame})

        if len(self.agent_widgets) > 1:
            sep = QWidget(); sep.setFixedHeight(2); sep.setStyleSheet("background:#666;")
            self.scroll_layout.addWidget(sep)
        self.scroll_layout.addWidget(frame)

    def get_crew_data(self):
        name = self.name_edit.text().strip()
        if not name: return None, None
        config = []
        for a in self.agent_widgets:
            role = a['role'].text().strip()
            inp = a['input'].toPlainText().strip()
            if role and inp:
                config.append({
                    'role': role,
                    'model': a['model'].currentText(),
                    'system_prompt': a['system'].toPlainText().strip(),
                    'input_prompt': inp
                })
        return name, config


# ================= MAIN GUI =================
class OllamaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BOFFIN • Local AI (Multi-Crew Manager)")
        self.resize(1700, 900)

        self.db = PostgresDB()
        self.current_conversation_id = None
        self.thread = None
        self.dark = True
        self.current_assistant_reply = ""
        self.advanced_mode = False
        self.current_crew_id = None
        self.current_crew_config = self.db.get_default_crew_config() or []
        self.current_crew_name = None
        self.models = []
        self.last_user_prompt = ""

        self.attached_image_path = None
        self.attached_image_base64 = None

        self.init_ui()
        self.load_models()
        self.refresh_conversations()
        self.refresh_crews_list()
        self.update_current_crew_button()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # LEFT SIDEBAR
        left = QVBoxLayout()
        left.setSpacing(10)

        left.addWidget(QLabel("<b>📂 Chats</b>"))
        self.new_chat_btn = QPushButton("➕ New Chat")
        self.new_chat_btn.clicked.connect(self.new_chat)
        left.addWidget(self.new_chat_btn)

        # সার্চ বার
        self.chat_search = QLineEdit()
        self.chat_search.setPlaceholderText("🔍 Search chats...")
        self.chat_search.setClearButtonEnabled(True)
        self.chat_search.textChanged.connect(self.filter_conversations)
        left.addWidget(self.chat_search)

        self.conv_list = QListWidget()
        self.conv_list.itemClicked.connect(self.load_conversation)
        self.conv_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.conv_list.customContextMenuRequested.connect(self.show_conv_menu)
        left.addWidget(self.conv_list)

        left.addWidget(QLabel("<b>⚙️ Crews</b>"))

        self.new_crew_btn = QPushButton("➕ Add New Crew")
        self.new_crew_btn.clicked.connect(self.create_new_crew)
        left.addWidget(self.new_crew_btn)

        self.crew_list = QListWidget()
        self.crew_list.itemClicked.connect(self.select_crew_from_list)
        self.crew_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.crew_list.customContextMenuRequested.connect(self.show_crew_menu)
        left.addWidget(self.crew_list, 1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(320)
        main.addWidget(left_widget)

        # RIGHT SIDE
        right = QVBoxLayout()
        top = QHBoxLayout()

        self.model_box = QComboBox()
        self.model_box.setMinimumWidth(400)

        self.mode_btn = QPushButton("⚡ Crew Mode: OFF")
        self.mode_btn.clicked.connect(self.toggle_advanced_mode)

        self.current_crew_btn = QPushButton("📋 No Crew Selected")
        self.current_crew_btn.clicked.connect(self.open_current_crew)

        self.theme_btn = QPushButton("🌙 Theme")
        self.theme_btn.clicked.connect(self.toggle_theme)

        top.addWidget(QLabel("Single Model:"))
        top.addWidget(self.model_box)
        top.addStretch()
        top.addWidget(self.mode_btn)
        top.addWidget(self.current_crew_btn)
        top.addWidget(self.theme_btn)
        right.addLayout(top)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setFont(QFont("DejaVu Sans", 26))
        right.addWidget(self.chat, 1)

        # Input + Attach
        input_layout = QHBoxLayout()
        self.attach_btn = QPushButton("📎")
        self.attach_btn.setFixedWidth(50)
        self.attach_btn.setToolTip("Attach Image")
        self.attach_btn.clicked.connect(self.attach_image)

        self.input = QTextEdit()
        self.input.setFixedHeight(120)
        self.input.setFont(QFont("DejaVu Sans", 26))

        input_layout.addWidget(self.attach_btn)
        input_layout.addWidget(self.input, 1)
        right.addLayout(input_layout)

        btns = QHBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send)
        self.send_btn.setMinimumHeight(40)

        self.stop_reload_btn = QPushButton("⏹ Stop")
        self.stop_reload_btn.clicked.connect(self.stop_or_reload)
        self.stop_reload_btn.setMinimumHeight(40)
        self.stop_reload_btn.setEnabled(False)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.chat.clear)
        self.clear_btn.setMinimumHeight(40)

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_chat)
        self.export_btn.setMinimumHeight(40)

        btns.addWidget(self.send_btn)
        btns.addWidget(self.stop_reload_btn)
        btns.addWidget(self.clear_btn)
        btns.addWidget(self.export_btn)
        right.addLayout(btns)

        main.addLayout(right)
        self.apply_theme()

    # ========== সার্চ ফিল্টার ==========
    def filter_conversations(self):
        search_text = self.chat_search.text().strip().lower()
        self.conv_list.clear()

        all_convos = self.db.list_conversations()
        sorted_convos = sorted(
            all_convos,
            key=lambda x: (not x.get("pinned", False), -x.get("id", 0))
        )

        for c in sorted_convos:
            title = (c["title"] or f"Chat {c['id']}").lower()
            if search_text == "" or search_text in title:
                prefix = "📌 " if c.get("pinned") else ""
                display_title = prefix + (c["title"] or f"Chat {c['id']}")
                item = QListWidgetItem(display_title)
                item.setData(Qt.UserRole, c["id"])
                self.conv_list.addItem(item)

    def refresh_conversations(self):
        self.filter_conversations()

    def stop_or_reload(self):
        if self.thread and self.thread.isRunning():
            if hasattr(self.thread, 'stop'):
                self.thread.stop()
            self.chat.append("\n⚠️ Generation stopped by user.\n")
            self.update_stop_reload_button(is_running=False)
        else:
            if self.last_user_prompt and self.current_conversation_id:
                self.input.setPlainText(self.last_user_prompt)
                self.send()
            else:
                self.chat.append("\nℹ️ No previous message to reload.\n")

    def update_stop_reload_button(self, is_running):
        if is_running:
            self.stop_reload_btn.setText("⏹ Stop")
            self.stop_reload_btn.setStyleSheet("QPushButton { background-color: #c33; color: white; font-weight: bold; }")
        else:
            self.stop_reload_btn.setText("🔄 Reload")
            self.stop_reload_btn.setStyleSheet("QPushButton { background-color: #393; color: white; }")
        self.stop_reload_btn.setEnabled(is_running or bool(self.last_user_prompt))

    def update_current_crew_button(self):
        if self.current_crew_config and self.current_crew_name:
            self.current_crew_btn.setText(f"📋 {self.current_crew_name} ({len(self.current_crew_config)} agents)")
            self.current_crew_btn.setStyleSheet("QPushButton { background-color: #2d8; color: white; font-weight: bold; }")
        else:
            self.current_crew_btn.setText("📋 No Crew Selected")
            self.current_crew_btn.setStyleSheet("")

    def load_models(self):
        self.models = []
        self.model_box.clear()
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if r.status_code != 200:
                raise Exception("Ollama server not responding")
            models_data = r.json().get("models", [])
            if not models_data:
                raise Exception("No models pulled")
            for m in models_data:
                name = m["name"]
                self.models.append(name)
                self.model_box.addItem(name)
        except Exception as e:
            fallback = "qwen2.5:latest"
            self.models.append(fallback)
            self.model_box.addItem(fallback + " (fallback)")
            self.chat.append(f"\n⚠️ <b>Ollama connection failed:</b> {str(e)}\n")
            self.chat.append("🔄 Using fallback model. Please start Ollama server and restart BOFFIN.\n")

    # ================= CREW METHODS =================
    def refresh_crews_list(self):
        self.crew_list.clear()
        crews = self.db.list_crews()
        for crew in crews:
            prefix = "⭐ " if crew['is_default'] else ""
            item = QListWidgetItem(f"{prefix}{crew['name']} ({len(json.loads(crew['config']))} agents)")
            item.setData(Qt.UserRole, crew['id'])
            self.crew_list.addItem(item)
        self.update_current_crew_button()

    def select_crew_from_list(self, item):
        crew_id = item.data(Qt.UserRole)
        crew = self.db.get_crew(crew_id)
        self.current_crew_config = json.loads(crew['config'])
        self.current_crew_id = crew_id
        self.current_crew_name = crew['name']
        self.update_current_crew_button()
        self.chat.append(f"\n✅ Switched to crew: <b>{crew['name']}</b> ({len(self.current_crew_config)} agents)\n")

    def show_crew_menu(self, pos):
        item = self.crew_list.itemAt(pos)
        menu = QMenu()
        if item:
            crew_id = item.data(Qt.UserRole)
            crew = self.db.get_crew(crew_id)
            menu.addAction("⭐ Set as Default").triggered.connect(lambda: self.set_default_crew(crew_id))
            rename_act = menu.addAction("✏ Rename")
            menu.addAction("✏ Edit").triggered.connect(lambda: self.edit_crew(crew_id))
            menu.addAction("🗑 Delete").triggered.connect(lambda: self.delete_crew(crew_id))
            action = menu.exec_(self.crew_list.mapToGlobal(pos))
            if action == rename_act:
                new_name, ok = QInputDialog.getText(self, "Rename Crew", "New crew name:", text=crew['name'])
                if ok and new_name.strip() and new_name.strip() != crew['name']:
                    self.db.update_crew_name(crew_id, new_name.strip())
                    self.refresh_crews_list()
                    if crew_id == self.current_crew_id:
                        self.current_crew_name = new_name.strip()
                        self.update_current_crew_button()
        else:
            menu.addAction("➕ Create New Crew").triggered.connect(self.create_new_crew)
            menu.exec_(self.crew_list.mapToGlobal(pos))

    def create_new_crew(self):
        dialog = CrewConfigDialog(self.models, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            name, config = dialog.get_crew_data()
            if name and config:
                new_id = self.db.create_crew(name, config)
                self.refresh_crews_list()
                self.current_crew_id = new_id
                self.current_crew_name = name
                self.current_crew_config = config
                self.update_current_crew_button()
                self.chat.append(f"\n✅ Created & switched to new crew: <b>{name}</b>\n")

    def edit_crew(self, crew_id):
        crew = self.db.get_crew(crew_id)
        config = json.loads(crew['config'])
        dialog = CrewConfigDialog(self.models, config, crew['name'], self)
        if dialog.exec_() == QDialog.Accepted:
            name, new_config = dialog.get_crew_data()
            if name and new_config:
                self.db.update_crew(crew_id, name, new_config)
                if crew_id == self.current_crew_id:
                    self.current_crew_config = new_config
                    self.current_crew_name = name
                    self.update_current_crew_button()
                self.refresh_crews_list()
                self.chat.append(f"\n✅ Updated crew: <b>{name}</b>\n")

    def delete_crew(self, crew_id):
        crew = self.db.get_crew(crew_id)
        reply = QMessageBox.question(self, "Delete Crew", f"Delete crew '{crew['name']}'?")
        if reply == QMessageBox.Yes:
            self.db.delete_crew(crew_id)
            if crew_id == self.current_crew_id:
                self.current_crew_config = []
                self.current_crew_id = None
                self.current_crew_name = None
                self.update_current_crew_button()
            self.refresh_crews_list()

    def set_default_crew(self, crew_id):
        self.db.set_default_crew(crew_id)
        crew = self.db.get_crew(crew_id)
        self.current_crew_id = crew_id
        self.current_crew_config = json.loads(crew['config'])
        self.current_crew_name = crew['name']
        self.update_current_crew_button()
        self.refresh_crews_list()
        self.chat.append(f"\n⭐ Set <b>{crew['name']}</b> as default crew!\n")

    def open_current_crew(self):
        if not self.current_crew_config:
            self.create_new_crew()
        else:
            if self.current_crew_id:
                self.edit_crew(self.current_crew_id)

    def toggle_advanced_mode(self):
        self.advanced_mode = not self.advanced_mode
        self.mode_btn.setText(f"⚡ Crew Mode: {'ON' if self.advanced_mode else 'OFF'}")

    def attach_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)"
        )
        if not path:
            return

        self.input.clear()
        cursor = self.input.textCursor()
        cursor.insertHtml(f'<img src="{path}" width="400" /><br>')
        self.input.setTextCursor(cursor)

        self.attached_image_path = path
        with open(path, "rb") as f:
            self.attached_image_base64 = base64.b64encode(f.read()).decode('utf-8')

        self.chat.append("\n📎 Image attached – will be sent with the next message.\n")

    def send(self):
        prompt = self.input.toPlainText().strip()
        if not prompt:
            return

        self.last_user_prompt = prompt
        self.input.clear()

        if not self.current_conversation_id:
            title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            self.current_conversation_id = self.db.create_conversation(title=title)
            self.refresh_conversations()

        self.db.add_message(self.current_conversation_id, "user", prompt)

        self.chat.append(f"\n🧑 YOU:\n{prompt}\n")
        if self.attached_image_path:
            cursor = self.chat.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertHtml(f'<br><img src="{self.attached_image_path}" width="500" /><br><br>')
            self.chat.setTextCursor(cursor)

        mode_text = f" (Crew: {len(self.current_crew_config)} agents)" if self.advanced_mode and self.current_crew_config else ""
        self.chat.append(f"🤖 BOFFIN{mode_text}:\n")
        self.current_assistant_reply = ""

        self.update_stop_reload_button(is_running=True)

        history = self.db.get_messages(self.current_conversation_id)
        ollama_messages = []
        for msg in history:
            m = {"role": msg["role"], "content": msg["content"]}
            if msg["role"] == "user" and msg == history[-1] and self.attached_image_base64:
                m["images"] = [self.attached_image_base64]
            ollama_messages.append(m)

        if self.advanced_mode:
            if not self.current_crew_config:
                self.chat.append("\n❌ No crew selected!\n")
                self.update_stop_reload_button(is_running=False)
                return
            self.thread = CustomCrewThread(prompt, self.current_crew_config, ollama_messages[:-1])
        else:
            model = self.model_box.currentText().split(" (")[0]
            self.thread = DirectOllamaThread(model, ollama_messages)

        self.thread.token.connect(self.append_token)
        self.thread.finished.connect(self.on_generation_finished)
        self.thread.error.connect(self.show_error)
        self.thread.start()

        self.attached_image_path = None
        self.attached_image_base64 = None

    def on_generation_finished(self, final=""):
        self.save_assistant_reply(final)
        self.update_stop_reload_button(is_running=False)

    def append_token(self, text):
        self.current_assistant_reply += text
        cursor = self.chat.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.chat.setTextCursor(cursor)
        self.chat.ensureCursorVisible()
        
        scrollbar = self.chat.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def save_assistant_reply(self, final=""):
        reply = final or self.current_assistant_reply.strip()
        if reply:
            self.db.add_message(self.current_conversation_id, "assistant", reply)
        self.current_assistant_reply = ""

    def show_error(self, e):
        self.chat.append(f"\n❌ Error: {e}\n")
        self.update_stop_reload_button(is_running=False)

    def show_conv_menu(self, pos):
        item = self.conv_list.itemAt(pos)
        if not item: return
        cid = item.data(Qt.UserRole)
        menu = QMenu()
        rename_act = menu.addAction("✏ Rename")
        pin_act = menu.addAction("📌 Pin / Unpin")
        delete_act = menu.addAction("🗑 Delete")
        action = menu.exec_(self.conv_list.mapToGlobal(pos))

        if action == rename_act:
            text, ok = QInputDialog.getText(self, "Rename Chat", "New title:")
            if ok and text.strip():
                self.db.rename_conversation(cid, text.strip())
                self.refresh_conversations()
        elif action == pin_act:
            self.db.toggle_pin(cid)
            self.refresh_conversations()
        elif action == delete_act:
            self.db.delete_conversation(cid)
            self.new_chat()
            self.refresh_conversations()

    def load_conversation(self, item):
        self.current_conversation_id = item.data(Qt.UserRole)
        self.chat.clear()
        messages = self.db.get_messages(self.current_conversation_id)
        for m in messages:
            who = "🧑 YOU" if m["role"] == "user" else "🤖 BOFFIN"
            self.chat.append(f"{who}:\n{m['content']}\n")
        if messages and messages[-1]["role"] == "user":
            self.last_user_prompt = messages[-1]["content"]
        self.update_stop_reload_button(is_running=False)

    def new_chat(self):
        self.current_conversation_id = None
        self.last_user_prompt = ""
        self.chat.clear()
        self.update_stop_reload_button(is_running=False)

    def export_chat(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Chat", "chat.md", "Markdown (*.md);;Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.chat.toPlainText())

    def toggle_theme(self):
        self.dark = not self.dark
        self.apply_theme()

    def apply_theme(self):
        if self.dark:
            self.setStyleSheet("""
                QMainWindow, QWidget { background:#121212; color:#e0e0e0; }
                QTextEdit { background:#1e1e1e; color:#fff; }
                QListWidget { background:#181818; color:#ddd; }
                QPushButton { background:#333; color:white; border:none; padding:8px; border-radius:6px; }
                QPushButton:hover { background:#555; }
                QComboBox { background:#222; color:white; }
                QLineEdit { background:#222; color:white; padding:6px; border-radius:6px; }
            """)
        else:
            self.setStyleSheet("")


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    win = OllamaGUI()
    win.show()
    sys.exit(app.exec_())
