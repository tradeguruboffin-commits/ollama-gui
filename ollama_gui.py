#!/usr/bin/env python3
import sys
import json
import requests
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication
from PyQt5.QtGui import QTextCursor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QComboBox,
    QLabel, QFileDialog, QListWidget,
    QListWidgetItem, QMenu, QInputDialog
)

from database.postgres import PostgresDB


# ================= DIRECT OLLAMA CALL THREAD =================

class DirectOllamaThread(QThread):
    token = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, model, prompt, system_prompt=""):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        try:
            url = "http://localhost:11434/api/generate"
            full_prompt = self.system_prompt + "\n\n" + self.prompt if self.system_prompt else self.prompt
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": True,
                "options": {"temperature": 0.7}
            }

            full_response = ""
            with requests.post(url, json=payload, stream=True, timeout=300) as r:
                for line in r.iter_lines():
                    if not self.running:
                        break
                    if line:
                        data = json.loads(line.decode())
                        if "response" in data:
                            token = data["response"]
                            full_response += token
                            self.token.emit(token)
                        if data.get("done"):
                            break

            self.finished.emit(full_response.strip())

        except Exception as e:
            self.error.emit(str(e))


class CodingCrewThread(QThread):
    token = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, user_prompt):
        super().__init__()
        self.user_prompt = user_prompt

    def run(self):
        try:
            full_output = "# BOFFIN CODING CREW REPORT\n\n"
            full_output += f"**User Request:** {self.user_prompt}\n\n"
            full_output += "---\n\n"

            # 1. Manager (llama3)
            self.token.emit("\n[Manager Planning...]\n")
            manager_thread = DirectOllamaThread(
                model="llama3:latest",
                system_prompt="You are a Project Manager. Create a detailed step-by-step plan for the user's project idea.",
                prompt=self.user_prompt
            )
            manager_thread.token.connect(self.token)
            manager_thread.start()
            manager_thread.wait()
            plan = manager_thread.property("result") if hasattr(manager_thread, "result") else "Plan generated."
            full_output += f"### Project Plan\n{plan}\n\n---\n\n"

            # 2. Coder (qwen2.5)
            self.token.emit("\n[Coder Writing Code...]\n")
            coder_thread = DirectOllamaThread(
                model="qwen2.5:latest",
                system_prompt="You are an expert Python coder. Write clean, complete, runnable code based on the plan. Include comments.",
                prompt=f"Project Plan:\n{plan}\n\nNow write the full Python code."
            )
            coder_thread.token.connect(self.token)
            coder_thread.start()
            coder_thread.wait()
            code = coder_thread.property("result") if hasattr(coder_thread, "result") else "Code generated."
            full_output += f"### Python Code\n```python\n{code}\n```\n\n---\n\n"

            # 3. Tester (phi3)
            self.token.emit("\n[Tester Reviewing...]\n")
            tester_thread = DirectOllamaThread(
                model="phi3:mini",
                system_prompt="You are a QA Tester. Review the code, find bugs, suggest improvements.",
                prompt=f"Code:\n```python\n{code}\n```\n\nTest it mentally and report issues or confirm it's good."
            )
            tester_thread.token.connect(self.token)
            tester_thread.start()
            tester_thread.wait()
            test_report = tester_thread.property("result") if hasattr(tester_thread, "result") else "Test passed."
            full_output += f"### Test Report\n{test_report}\n\n---\n\n"

            # 4. Documentation (tinyllama)
            self.token.emit("\n[Writing Documentation...]\n")
            doc_thread = DirectOllamaThread(
                model="tinyllama:latest",
                system_prompt="You are a technical writer. Write a professional README.md.",
                prompt=f"Project: {self.user_prompt}\nPlan: {plan}\nFinal Code: {code}\n\nWrite a complete README.md."
            )
            doc_thread.token.connect(self.token)
            doc_thread.start()
            doc_thread.wait()
            readme = doc_thread.property("result") if hasattr(doc_thread, "result") else "README generated."
            full_output += f"### README.md\n{readme}\n"

            self.finished.emit(full_output)

        except Exception as e:
            self.error.emit(str(e))


# ================= MAIN GUI =================

class OllamaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BOFFIN • Ollama AI (Local Coding Crew)")
        self.resize(1700, 900)

        self.db = PostgresDB()
        self.current_conversation_id = None
        self.thread = None
        self.dark = True
        self.current_assistant_reply = ""
        self.advanced_mode = False

        self.init_ui()
        self.load_models()
        self.refresh_conversations()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        left = QVBoxLayout()
        self.new_chat_btn = QPushButton("➕ New Chat")
        self.new_chat_btn.setMinimumHeight(42)
        self.new_chat_btn.clicked.connect(self.new_chat)

        self.conv_list = QListWidget()
        self.conv_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.conv_list.customContextMenuRequested.connect(self.show_conv_menu)
        self.conv_list.itemClicked.connect(self.load_conversation)

        left.addWidget(self.new_chat_btn)
        left.addWidget(self.conv_list)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(320)
        main.addWidget(left_widget)

        right = QVBoxLayout()
        main.addLayout(right)

        top = QHBoxLayout()
        self.model_box = QComboBox()
        self.model_box.setMinimumWidth(400)

        self.mode_btn = QPushButton("⚡ Coding Crew Mode: OFF")
        self.mode_btn.clicked.connect(self.toggle_advanced_mode)

        self.theme_btn = QPushButton("🌙 Theme")
        self.theme_btn.clicked.connect(self.toggle_theme)

        top.addWidget(QLabel("Single Model:"))
        top.addWidget(self.model_box)
        top.addStretch()
        top.addWidget(self.mode_btn)
        top.addWidget(self.theme_btn)
        right.addLayout(top)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setFont(QFont("DejaVu Sans", 26))
        right.addWidget(self.chat, 1)

        self.input = QTextEdit()
        self.input.setFixedHeight(120)
        self.input.setFont(QFont("DejaVu Sans", 26))
        right.addWidget(self.input)

        btns = QHBoxLayout()
        self.send_btn = QPushButton("Send")
        self.stop_btn = QPushButton("Stop")
        self.clear_btn = QPushButton("Clear")
        self.export_btn = QPushButton("Export")

        self.send_btn.clicked.connect(self.send)
        self.stop_btn.clicked.connect(self.stop)
        self.clear_btn.clicked.connect(self.chat.clear)
        self.export_btn.clicked.connect(self.export_chat)

        for b in (self.send_btn, self.stop_btn, self.clear_btn, self.export_btn):
            b.setMinimumHeight(40)
            btns.addWidget(b)

        right.addLayout(btns)
        self.apply_theme()

    def toggle_advanced_mode(self):
        self.advanced_mode = not self.advanced_mode
        status = "ON" if self.advanced_mode else "OFF"
        self.mode_btn.setText(f"⚡ Coding Crew Mode: {status}")

    def load_models(self):
        self.model_box.clear()
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            for m in r.json().get("models", []):
                self.model_box.addItem(m["name"])
        except Exception:
            self.model_box.addItem("qwen2.5:latest")

    def new_chat(self):
        self.current_conversation_id = None
        self.chat.clear()

    def send(self):
        prompt = self.input.toPlainText().strip()
        if not prompt:
            return

        model = self.model_box.currentText()
        self.input.clear()

        if self.current_conversation_id is None:
            title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            self.current_conversation_id = self.db.create_conversation(title=title, model=model)
            self.refresh_conversations()

        self.db.add_message(self.current_conversation_id, "user", prompt)
        mode_text = " (Local Coding Crew)" if self.advanced_mode else ""
        self.chat.append(f"\n🧑 YOU:\n{prompt}\n\n🤖 BOFFIN{mode_text}:\n")
        self.current_assistant_reply = ""

        if self.advanced_mode:
            self.thread = CodingCrewThread(prompt)
        else:
            self.thread = DirectOllamaThread(model, prompt)

        self.thread.token.connect(self.append_token)
        self.thread.finished.connect(self.save_assistant_reply)
        self.thread.error.connect(self.show_error)
        self.thread.start()

    def append_token(self, text):
        self.current_assistant_reply += text
        cursor = self.chat.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.chat.setTextCursor(cursor)
        self.chat.ensureCursorVisible()

    def save_assistant_reply(self, final=""):
        reply = final if final else self.current_assistant_reply.strip()
        if reply:
            self.db.add_message(self.current_conversation_id, "assistant", reply)

    def stop(self):
        if self.thread and self.thread.isRunning():
            if hasattr(self.thread, 'stop'):
                self.thread.stop()
            self.thread.terminate()

    # বাকি ফাংশনগুলো (refresh_conversations, load_conversation, etc.) আগের মতোই রাখো

    def refresh_conversations(self):
        self.conv_list.clear()
        conversations = self.db.list_conversations()
        conversations = sorted(conversations, key=lambda x: (not x.get("pinned", False), x["created_at"]), reverse=True)
        for c in conversations:
            title = c["title"] or f"Chat {c['id']}"
            if c.get("pinned"):
                title = "📌 " + title
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, c["id"])
            self.conv_list.addItem(item)

    def load_conversation(self, item):
        cid = item.data(Qt.UserRole)
        self.current_conversation_id = cid
        self.chat.clear()
        for m in self.db.get_messages(cid):
            who = "🧑 YOU" if m["role"] == "user" else "🤖 BOFFIN"
            self.chat.append(f"{who}:\n{m['content']}\n")

    def show_conv_menu(self, pos):
        item = self.conv_list.itemAt(pos)
        if not item:
            return
        cid = item.data(Qt.UserRole)
        menu = QMenu(self)
        rename = menu.addAction("✏ Rename")
        pin = menu.addAction("📌 Pin / Unpin")
        delete = menu.addAction("🗑 Delete")
        action = menu.exec_(self.conv_list.mapToGlobal(pos))
        if action == rename:
            text, ok = QInputDialog.getText(self, "Rename", "New name:")
            if ok and text.strip():
                self.db.rename_conversation(cid, text.strip())
                self.refresh_conversations()
        elif action == pin:
            self.db.toggle_pin(cid)
            self.refresh_conversations()
        elif action == delete:
            self.db.delete_conversation(cid)
            self.new_chat()
            self.refresh_conversations()

    def export_chat(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Chat", "chat.txt")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.chat.toPlainText())

    def show_error(self, e):
        self.chat.append(f"\n❌ Error: {e}")

    def toggle_theme(self):
        self.dark = not self.dark
        self.apply_theme()

    def apply_theme(self):
        if self.dark:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #121212; color: #e0e0e0; }
                QTextEdit { background-color: #1e1e1e; color: #ffffff; }
                QListWidget { background-color: #181818; color: #bbb; }
                QPushButton { background-color: #333; color: white; }
                QComboBox { background-color: #222; color: white; }
            """)
        else:
            self.setStyleSheet("")


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    win = OllamaGUI()
    win.show()
    sys.exit(app.exec_())
