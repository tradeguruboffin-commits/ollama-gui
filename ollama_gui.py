#!/usr/bin/env python3
import sys
import json
import requests
import base64
import os
import subprocess
import glob
import shutil
import time
import mimetypes
import copy
import hashlib
import threading
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker, QTimer
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QComboBox, QLabel, QFileDialog,
    QListWidget, QListWidgetItem, QMenu, QInputDialog,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QScrollArea, QMessageBox, QProgressBar
)

try:
    from database.postgres import PostgresDB
    DB_CLASS = PostgresDB
except Exception:
    import sqlite3

    class SQLiteDB:
        def __init__(self):
            db_dir = os.path.join(os.path.expanduser("~"), ".ollama_gui")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "chat.db")
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.lock = threading.Lock()
            self.create_tables()

        def create_tables(self):
            with self.lock:
                c = self.conn.cursor()
                c.execute('''CREATE TABLE IF NOT EXISTS conversations
                             (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, pinned INTEGER DEFAULT 0)''')
                c.execute('''CREATE TABLE IF NOT EXISTS messages
                             (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER,
                              role TEXT, content TEXT)''')
                c.execute('''CREATE TABLE IF NOT EXISTS crews
                             (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE,
                              config TEXT, is_default INTEGER DEFAULT 0)''')
                self.conn.commit()

        def create_conversation(self, title=None):
            with self.lock:
                c = self.conn.cursor()
                c.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
                self.conn.commit()
                return c.lastrowid

        def list_conversations(self):
            with self.lock:
                c = self.conn.cursor()
                c.execute("SELECT id, title, pinned FROM conversations ORDER BY id DESC")
                return [{"id": r[0], "title": r[1], "pinned": bool(r[2])} for r in c.fetchall()]

        def rename_conversation(self, cid, title):
            with self.lock:
                c = self.conn.cursor()
                c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, cid))
                self.conn.commit()

        def toggle_pin(self, cid):
            with self.lock:
                c = self.conn.cursor()
                c.execute("UPDATE conversations SET pinned = NOT pinned WHERE id = ?", (cid,))
                self.conn.commit()

        def delete_conversation(self, cid):
            with self.lock:
                c = self.conn.cursor()
                c.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
                c.execute("DELETE FROM conversations WHERE id = ?", (cid,))
                self.conn.commit()

        def add_message(self, cid, role, content):
            with self.lock:
                c = self.conn.cursor()
                c.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                          (cid, role, content))
                self.conn.commit()

        def get_messages(self, cid):
            with self.lock:
                c = self.conn.cursor()
                c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC", (cid,))
                return [{"role": r[0], "content": r[1]} for r in c.fetchall()]

        def create_crew(self, name, config):
            with self.lock:
                c = self.conn.cursor()
                c.execute("INSERT INTO crews (name, config) VALUES (?, ?)", (name, json.dumps(config)))
                self.conn.commit()
                return c.lastrowid

        def list_crews(self):
            with self.lock:
                c = self.conn.cursor()
                c.execute("SELECT id, name, config, is_default FROM crews")
                return [{"id": r[0], "name": r[1], "config": r[2], "is_default": bool(r[3])} for r in c.fetchall()]

        def get_crew(self, crew_id):
            with self.lock:
                c = self.conn.cursor()
                c.execute("SELECT name, config FROM crews WHERE id = ?", (crew_id,))
                row = c.fetchone()
                return {"name": row[0], "config": row[1]} if row else None

        def update_crew(self, crew_id, name, config):
            with self.lock:
                c = self.conn.cursor()
                c.execute("UPDATE crews SET name = ?, config = ? WHERE id = ?", (name, json.dumps(config), crew_id))
                self.conn.commit()

        def update_crew_name(self, crew_id, name):
            with self.lock:
                c = self.conn.cursor()
                c.execute("UPDATE crews SET name = ? WHERE id = ?", (name, crew_id))
                self.conn.commit()

        def delete_crew(self, crew_id):
            with self.lock:
                c = self.conn.cursor()
                c.execute("DELETE FROM crews WHERE id = ?", (crew_id,))
                self.conn.commit()

        def set_default_crew(self, crew_id):
            with self.lock:
                c = self.conn.cursor()
                c.execute("UPDATE crews SET is_default = 0")
                c.execute("UPDATE crews SET is_default = 1 WHERE id = ?", (crew_id,))
                self.conn.commit()

        def get_default_crew_config(self):
            with self.lock:
                c = self.conn.cursor()
                c.execute("SELECT config FROM crews WHERE is_default = 1 LIMIT 1")
                row = c.fetchone()
                return json.loads(row[0]) if row else None

    DB_CLASS = SQLiteDB

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

class DirectOllamaThread(QThread):
    token = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(str, float, int)

    def __init__(self, model, messages):
        super().__init__()
        self.model = model
        self.messages = messages
        self.mutex = QMutex()
        self._running = True
        self.start_time = None
        self.chunk_count = 0
        self.buffer = ""
        self.last_flush = 0

    def stop(self):
        with QMutexLocker(self.mutex):
            self._running = False

    def is_running(self):
        with QMutexLocker(self.mutex):
            return self._running

    def run(self):
        self.start_time = time.time()
        response = ""
        try:
            url = "http://localhost:11434/api/chat"
            payload = {"model": self.model, "messages": self.messages, "stream": True, "options": {"temperature": 0.7}}
            with requests.post(url, json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not self.is_running():
                        r.close()
                        break
                    if line:
                        try:
                            data = json.loads(line.decode())
                        except json.JSONDecodeError:
                            continue
                        if "message" in data and "content" in data["message"]:
                            token = data["message"]["content"]
                            response += token
                            self.chunk_count += 1
                            self.buffer += token
                            now = time.time()
                            if now - self.last_flush > 0.05:
                                self.token.emit(self.buffer)
                                self.buffer = ""
                                self.last_flush = now
                        if data.get("done"):
                            break
            if self.buffer:
                self.token.emit(self.buffer)
            elapsed = time.time() - self.start_time
            self.finished.emit(response.strip(), elapsed, self.chunk_count)
        except Exception as e:
            self.error.emit(str(e))

class CustomCrewThread(QThread):
    token = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(str, float, int)

    def __init__(self, user_prompt, crew_config, history_messages):
        super().__init__()
        self.user_prompt = user_prompt
        self.crew_config = crew_config
        self.history_messages = history_messages
        self.mutex = QMutex()
        self._running = True
        self.start_time = None
        self.total_chunks = 0
        self.buffer = ""
        self.last_flush = 0

    def stop(self):
        with QMutexLocker(self.mutex):
            self._running = False

    def is_running(self):
        with QMutexLocker(self.mutex):
            return self._running

    def run_agent_inline(self, model, messages):
        response = ""
        try:
            url = "http://localhost:11434/api/chat"
            payload = {"model": model, "messages": messages, "stream": True, "options": {"temperature": 0.7}}
            with requests.post(url, json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not self.is_running():
                        r.close()
                        break
                    if line:
                        try:
                            data = json.loads(line.decode())
                        except json.JSONDecodeError:
                            continue
                        if "message" in data and "content" in data["message"]:
                            token = data["message"]["content"]
                            response += token
                            self.total_chunks += 1
                            self.buffer += token
                            now = time.time()
                            if now - self.last_flush > 0.05:
                                self.token.emit(self.buffer)
                                self.buffer = ""
                                self.last_flush = now
        except Exception as e:
            self.error.emit(f"[ERROR in {model}: {str(e)}]")
        if self.buffer:
            self.token.emit(self.buffer)
        return response.strip()

    def run(self):
        self.start_time = time.time()
        output = "# OLLAMA CUSTOM CREW REPORT\n\n**User Request:** " + self.user_prompt + "\n\n---\n\n"
        previous = self.user_prompt

        for i, agent in enumerate(self.crew_config, 1):
            if not self.is_running():
                break
            role = agent['role']
            model = agent['model']
            system_prompt = agent.get('system_prompt', '').strip()
            input_prompt = agent['input_prompt'].format(previous=previous)

            self.token.emit(f"\n[👤 {i}. {role} ({model}) Working...]\n")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if i == 1 and self.history_messages:
                user_history = [m['content'] for m in self.history_messages if m['role'] == 'user'][-6:]
                if user_history:
                    input_prompt += "\n\nPrevious user messages:\n" + "\n".join(user_history)

            messages.append({"role": "user", "content": input_prompt})

            out = self.run_agent_inline(model, messages)
            if not out:
                out = "[No response]"
            output += f"### {role} Output\n{out}\n\n---\n\n"
            previous = out

        elapsed = time.time() - self.start_time
        final = output if self.is_running() else output + "\n\n[GENERATION STOPPED BY USER]"
        self.finished.emit(final, elapsed, self.total_chunks)

class RAGWorker(QThread):
    progress = pyqtSignal(int, int)
    message = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths
        self.mutex = QMutex()
        self._running = True

    def stop(self):
        with QMutexLocker(self.mutex):
            self._running = False

    def is_running(self):
        with QMutexLocker(self.mutex):
            return self._running

    def run(self):
        try:
            docs = []
            seen_hashes = set()
            for p in self.paths:
                if not self.is_running():
                    return
                try:
                    with open(p, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in seen_hashes:
                        self.message.emit(f"Skipped duplicate: {os.path.basename(p)}")
                        continue
                    seen_hashes.add(file_hash)

                    if p.lower().endswith(".pdf"):
                        loader = PyPDFLoader(p)
                    elif p.lower().endswith(".docx"):
                        loader = Docx2txtLoader(p)
                    elif p.lower().endswith(".md"):
                        loader = UnstructuredMarkdownLoader(p)
                    elif p.lower().endswith((".html", ".htm")):
                        from langchain_community.document_loaders import UnstructuredHTMLLoader
                        loader = UnstructuredHTMLLoader(p)
                    else:
                        loader = TextLoader(p, encoding="utf-8")

                    loaded = loader.load()
                    for d in loaded:
                        d.metadata["file_hash"] = file_hash
                        d.metadata["source"] = os.path.basename(p)
                    docs.extend(loaded)
                except Exception as e:
                    self.message.emit(f"Failed: {os.path.basename(p)} - {str(e)}")

            if not self.is_running():
                return

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            embed = OllamaEmbeddings(model="nomic-embed-text:latest")

            persist_dir = os.path.join(os.path.expanduser("~"), ".ollama_gui", "rag_db")
            os.makedirs(persist_dir, exist_ok=True)
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=embed, collection_name="rag_main")

            existing = vectordb.get(include=["metadatas"])
            existing_hashes = {m.get("file_hash") for m in existing["metadatas"] if m.get("file_hash")}
            new_chunks = [c for c in chunks if c.metadata.get("file_hash") not in existing_hashes]

            self.progress.emit(0, len(new_chunks))
            batch_size = 10
            for i in range(0, len(new_chunks), batch_size):
                if not self.is_running():
                    return
                batch = new_chunks[i:i+batch_size]
                vectordb.add_documents(batch)
                self.progress.emit(i + batch_size, len(new_chunks))

            self.finished.emit(vectordb.as_retriever(search_kwargs={"k": 5}))
        except Exception as e:
            self.error.emit(str(e))

CREW_TEMPLATES = [
    {"name": "Research Crew", "config": [
        {"role": "Researcher", "model": "llama3.2:latest", "system_prompt": "You are an expert researcher.", "input_prompt": "Research: {previous}"},
        {"role": "Analyzer", "model": "llama3.2:latest", "system_prompt": "Analyze findings deeply.", "input_prompt": "{previous}"},
        {"role": "Writer", "model": "llama3.2:latest", "system_prompt": "Write a professional report.", "input_prompt": "{previous}"}
    ]},
    {"name": "Coding Crew", "config": [
        {"role": "Architect", "model": "llama3.2:latest", "system_prompt": "Design the solution architecture.", "input_prompt": "Plan: {previous}"},
        {"role": "Coder", "model": "llama3.2:latest", "system_prompt": "Write clean, efficient code.", "input_prompt": "{previous}"},
        {"role": "Reviewer", "model": "llama3.2:latest", "system_prompt": "Review and suggest improvements.", "input_prompt": "{previous}"}
    ]},
    {"name": "Writing Crew", "config": [
        {"role": "Outliner", "model": "llama3.2:latest", "system_prompt": "Create a detailed outline.", "input_prompt": "Outline: {previous}"},
        {"role": "Drafter", "model": "llama3.2:latest", "system_prompt": "Write the first draft.", "input_prompt": "{previous}"},
        {"role": "Editor", "model": "llama3.2:latest", "system_prompt": "Edit for clarity and style.", "input_prompt": "{previous}"}
    ]}
]

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
        frame.setStyleSheet("background:#222;border:1px solid #555;border-radius:10px;padding:12px;margin:8px;")
        form = QFormLayout(frame)

        role = QLineEdit()
        role.setPlaceholderText("e.g. Researcher")
        model = QComboBox()
        model.addItems(self.models)
        system = QTextEdit()
        system.setFixedHeight(80)
        inp = QTextEdit()
        inp.setFixedHeight(130)
        inp.setPlaceholderText("Use {previous} to reference prior output")

        if preset:
            role.setText(preset.get('role', ''))
            model.setCurrentText(preset.get('model', ''))
            system.setPlainText(preset.get('system_prompt', ''))
            inp.setPlainText(preset.get('input_prompt', ''))

        form.addRow("👤 Role:", role)
        form.addRow("🤖 Model:", model)
        form.addRow("📝 System Prompt:", system)
        form.addRow("📋 Input Template:", inp)

        self.agent_widgets.append({'role': role, 'model': model, 'system': system, 'input': inp, 'frame': frame})

        if len(self.agent_widgets) > 1:
            sep = QWidget()
            sep.setFixedHeight(2)
            sep.setStyleSheet("background:#666;")
            self.scroll_layout.addWidget(sep)
        self.scroll_layout.addWidget(frame)

    def get_crew_data(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Crew name required")
            return None, None

        config = []
        for a in self.agent_widgets:
            role = a['role'].text().strip()
            inp = a['input'].toPlainText().strip()
            if not role:
                QMessageBox.warning(self, "Error", "All agents must have a role")
                return None, None
            if "{previous}" not in inp:
                reply = QMessageBox.question(self, "Warning", f"Agent '{role}' has no {{previous}}. Continue?")
                if reply == QMessageBox.No:
                    return None, None
            config.append({
                'role': role,
                'model': a['model'].currentText(),
                'system_prompt': a['system'].toPlainText().strip(),
                'input_prompt': inp
            })
        return name, config

class OllamaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OLLAMA • Local AI (Multi-Crew Manager)")
        self.resize(1700, 900)

        self.db = DB_CLASS()
        self.db_mutex = QMutex()
        self.current_conversation_id = None
        self.thread = None
        self.dark = True
        self.advanced_mode = False
        self.current_crew_id = None
        self.current_crew_config = self.db.get_default_crew_config() or []
        self.current_crew_name = None
        self.models = []
        self.last_user_prompt = ""

        self.attached_image_path = None
        self.attached_image_base64 = None

        self.retriever = None
        self.model_vision_cache = {}

        self.init_ui()
        self.load_models()
        self.refresh_conversations()
        self.refresh_crews_list()
        self.update_current_crew_button()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        left = QVBoxLayout()
        left.setSpacing(15)
        left.addWidget(QLabel("<b>📂 Chats</b>"))
        self.new_chat_btn = QPushButton("➕ New Chat")
        self.new_chat_btn.clicked.connect(self.new_chat)
        left.addWidget(self.new_chat_btn)

        self.chat_search = QLineEdit()
        self.chat_search.setPlaceholderText("🔍 Search chats...")
        self.chat_search.setClearButtonEnabled(True)
        self.chat_search.textChanged.connect(self.filter_conversations)
        left.addWidget(self.chat_search)

        self.conv_list = QListWidget()
        self.conv_list.itemClicked.connect(self.load_conversation)
        self.conv_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.conv_list.customContextMenuRequested.connect(self.show_conv_menu)
        left.addWidget(self.conv_list, 1)

        self.manager_btn = QPushButton("🦙 Manager")
        self.manager_btn.clicked.connect(self.open_model_manager)
        self.manager_btn.setStyleSheet("background:#1a5f1a;color:white;font-weight:bold;padding:15px;border-radius:35px;font-size:18px;min-height:70px;")
        left.addWidget(self.manager_btn, alignment=Qt.AlignCenter)

        self.rag_btn = QPushButton("📂 Add Knowledge (RAG)")
        self.rag_btn.clicked.connect(self.add_rag_knowledge)
        self.rag_btn.setStyleSheet("background:#2d5;color:white;font-weight:bold;padding:12px;border-radius:8px;")
        self.rag_btn.setMinimumHeight(50)
        left.addWidget(self.rag_btn)

        self.cancel_rag_btn = QPushButton("❌ Cancel RAG")
        self.cancel_rag_btn.clicked.connect(self.cancel_rag)
        self.cancel_rag_btn.setVisible(False)
        left.addWidget(self.cancel_rag_btn)

        self.clear_rag_btn = QPushButton("🗑️ Clear Knowledge")
        self.clear_rag_btn.clicked.connect(self.clear_rag_knowledge)
        self.clear_rag_btn.setEnabled(False)
        left.addWidget(self.clear_rag_btn)

        self.rag_progress = QProgressBar()
        self.rag_progress.setVisible(False)
        left.addWidget(self.rag_progress)

        left.addWidget(QLabel("<b>⚙️ Crews</b>"))
        self.new_crew_btn = QPushButton("➕ Add New Crew")
        self.new_crew_btn.clicked.connect(self.create_new_crew)
        left.addWidget(self.new_crew_btn)

        self.template_btn = QPushButton("📑 Load Template")
        self.template_btn.clicked.connect(self.load_crew_template)
        left.addWidget(self.template_btn)

        self.crew_list = QListWidget()
        self.crew_list.itemClicked.connect(self.select_crew_from_list)
        self.crew_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.crew_list.customContextMenuRequested.connect(self.show_crew_menu)
        left.addWidget(self.crew_list, 1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(320)
        main.addWidget(left_widget)

        right = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(QLabel("Single Model:"))
        self.model_box = QComboBox()
        self.model_box.setMinimumWidth(400)
        top.addWidget(self.model_box)
        top.addStretch()
        self.mode_btn = QPushButton("⚡ Crew Mode: OFF")
        self.mode_btn.clicked.connect(self.toggle_advanced_mode)
        top.addWidget(self.mode_btn)
        self.current_crew_btn = QPushButton("📋 No Crew Selected")
        self.current_crew_btn.clicked.connect(self.open_current_crew)
        top.addWidget(self.current_crew_btn)
        self.theme_btn = QPushButton("🌙 Theme")
        self.theme_btn.clicked.connect(self.toggle_theme)
        top.addWidget(self.theme_btn)
        right.addLayout(top)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setFont(QFont("DejaVu Sans", 26))
        right.addWidget(self.chat, 1)

        input_layout = QHBoxLayout()
        self.attach_btn = QPushButton("📎")
        self.attach_btn.clicked.connect(self.attach_image)
        input_layout.addWidget(self.attach_btn)
        self.input = QTextEdit()
        self.input.setFixedHeight(120)
        self.input.setFont(QFont("DejaVu Sans", 26))
        input_layout.addWidget(self.input, 1)
        right.addLayout(input_layout)

        btns = QHBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send)
        btns.addWidget(self.send_btn)
        self.stop_reload_btn = QPushButton("⏹ Stop")
        self.stop_reload_btn.clicked.connect(self.stop_or_reload)
        self.stop_reload_btn.setEnabled(False)
        btns.addWidget(self.stop_reload_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.chat.clear)
        btns.addWidget(self.clear_btn)
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_chat)
        btns.addWidget(self.export_btn)
        right.addLayout(btns)

        main.addLayout(right)
        self.apply_theme()

    def apply_theme(self):
        if self.dark:
            self.setStyleSheet("""
                QMainWindow, QWidget { background:#121212; color:#e0e0e0; }
                QTextEdit { background:#1e1e1e; color:#fff; border:none; }
                QListWidget { background:#181818; color:#ddd; border:none; padding:10px; font-size:20px; }
                QListWidget::item { padding:20px; min-height:60px; border-bottom:1px solid #2a2a2a; border-radius:12px; margin:6px 10px; }
                QListWidget::item:hover { background:#2a2a2a; }
                QListWidget::item:selected { background:#1f6feb; color:white; font-weight:bold; }
                QPushButton { background:#333; color:white; border:none; padding:12px; border-radius:8px; font-size:18px; }
                QPushButton:hover { background:#555; }
                QProgressBar { background:#222; color:white; border-radius:8px; }
                QProgressBar::chunk { background:#2d8; }
                QComboBox, QLineEdit { background:#222; color:white; padding:10px; border-radius:8px; font-size:18px; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget { background:#f8f9fa; color:#212529; }
                QTextEdit { background:#fff; color:#212529; border:1px solid #ced4da; }
                QListWidget { background:#fff; color:#212529; border:1px solid #dee2e6; }
                QPushButton { background:#0d6efd; color:white; }
                QPushButton:hover { background:#0b5ed7; }
                QProgressBar::chunk { background:#0d6efd; }
                QComboBox, QLineEdit { background:#fff; color:#212529; border:1px solid #ced4da; }
            """)

    def load_models(self):
        self.models = []
        self.model_box.clear()
        self.model_vision_cache.clear()
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            r.raise_for_status()
            for m in r.json()["models"]:
                name = m["name"]
                self.models.append(name)
                self.model_box.addItem(name)
                lower = name.lower()
                self.model_vision_cache[name] = any(k in lower for k in ("llava", "vision", "vl", "bakllava", "moondream", "phi3-v"))
        except Exception as e:
            fallback = "llama3.2:latest"
            self.model_box.addItem(fallback + " (fallback)")
            self.model_vision_cache[fallback] = False
            self.chat.append(f"\n⚠️ Ollama not reachable: {e}\nUsing fallback.\n")

    def attach_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif)")
        if not path:
            return
        self.input.clear()
        cursor = self.input.textCursor()
        cursor.insertHtml(f'<img src="{path}" width="400"><br>')
        self.input.setTextCursor(cursor)
        with open(path, "rb") as f:
            self.attached_image_base64 = base64.b64encode(f.read()).decode()
        self.attached_image_path = path
        self.chat.append("\n📎 Image attached.\n")

    def add_rag_knowledge(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Documents", "", "Documents (*.pdf *.txt *.md *.docx *.html)")
        folder = QFileDialog.getExistingDirectory(self, "Or Select Folder")
        paths = files + (glob.glob(os.path.join(folder, "**/*.*"), recursive=True) if folder else [])
        if not paths:
            return
        self.rag_btn.setEnabled(False)
        self.cancel_rag_btn.setVisible(True)
        self.rag_progress.setVisible(True)
        self.rag_progress.setValue(0)
        self.chat.append("\n🔄 Building knowledge base...\n")
        self.rag_worker = RAGWorker(paths)
        self.rag_worker.progress.connect(self.update_rag_progress)
        self.rag_worker.message.connect(lambda m: self.chat.append(m))
        self.rag_worker.finished.connect(self.on_rag_finished)
        self.rag_worker.error.connect(self.on_rag_error)
        self.rag_worker.start()

    def update_rag_progress(self, done, total):
        self.rag_progress.setMaximum(total)
        self.rag_progress.setValue(done)

    def cancel_rag(self):
        if hasattr(self, "rag_worker") and self.rag_worker.isRunning():
            self.rag_worker.stop()
            self.chat.append("\n⚠️ RAG cancelled.\n")
        self.cancel_rag_btn.setVisible(False)
        self.rag_progress.setVisible(False)
        self.rag_progress.reset()
        self.rag_btn.setEnabled(True)

    def on_rag_finished(self, retriever):
        self.retriever = retriever
        self.cancel_rag_btn.setVisible(False)
        self.rag_progress.setVisible(False)
        self.rag_btn.setEnabled(True)
        self.clear_rag_btn.setEnabled(True)
        self.chat.append("\n✅ Knowledge base ready!\n")

    def on_rag_error(self, err):
        self.chat.append(f"\n❌ RAG Error: {err}\n")
        QMessageBox.critical(self, "RAG Error", str(err))
        self.cancel_rag_btn.setVisible(False)
        self.rag_progress.setVisible(False)
        self.rag_btn.setEnabled(True)

    def clear_rag_knowledge(self):
        reply = QMessageBox.question(self, "Clear RAG", "This will delete all knowledge and require app restart.\nContinue?")
        if reply != QMessageBox.Yes:
            return
        persist_dir = os.path.join(os.path.expanduser("~"), ".ollama_gui", "rag_db")
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)
        self.retriever = None
        self.clear_rag_btn.setEnabled(False)
        QMessageBox.information(self, "Cleared", "RAG cleared. Restart the app.")
        QApplication.quit()

    def open_model_manager(self):
        manager_file = "ollama_manager.py"
        if not os.path.exists(manager_file):
            self.chat.append("\n❌ <b>ollama_manager.py</b> file not found!\n")
            QMessageBox.critical(self, "File Missing", f"Place <b>{manager_file}</b> in the same folder.")
            return
        try:
            subprocess.Popen([sys.executable, manager_file], cwd=os.path.dirname(__file__) or ".")
            self.chat.append("\n🦙 <b>Ollama Model Manager</b> opened!\n")
        except Exception as e:
            self.chat.append(f"\n❌ Failed: {str(e)}\n")

    def filter_conversations(self):
        search_text = self.chat_search.text().strip().lower()
        self.conv_list.clear()
        all_convos = self.db.list_conversations()
        sorted_convos = sorted(all_convos, key=lambda x: (not x.get("pinned", False), -x.get("id", 0)))
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
            self.update_stop_reload_button(False)
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

    def new_chat(self):
        self.current_conversation_id = None
        self.last_user_prompt = ""
        self.chat.clear()
        self.update_stop_reload_button(False)

    def load_conversation(self, item):
        self.current_conversation_id = item.data(Qt.UserRole)
        self.chat.clear()
        with QMutexLocker(self.db_mutex):
            messages = self.db.get_messages(self.current_conversation_id)
        for m in messages:
            who = "🧑 YOU" if m["role"] == "user" else "🤖 BOFFIN"
            self.chat.append(f"{who}:\n{m['content']}\n")
        if messages and messages[-1]["role"] == "user":
            self.last_user_prompt = messages[-1]["content"]
        self.update_stop_reload_button(False)

    def export_chat(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Chat", "chat.md", "Markdown (*.md);;Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.chat.toPlainText())

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
                with QMutexLocker(self.db_mutex):
                    self.db.rename_conversation(cid, text.strip())
                self.refresh_conversations()
        elif action == pin_act:
            with QMutexLocker(self.db_mutex):
                self.db.toggle_pin(cid)
            self.refresh_conversations()
        elif action == delete_act:
            with QMutexLocker(self.db_mutex):
                self.db.delete_conversation(cid)
            self.new_chat()
            self.refresh_conversations()

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
                    with QMutexLocker(self.db_mutex):
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
                with QMutexLocker(self.db_mutex):
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
                with QMutexLocker(self.db_mutex):
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
            with QMutexLocker(self.db_mutex):
                self.db.delete_crew(crew_id)
            if crew_id == self.current_crew_id:
                self.current_crew_config = []
                self.current_crew_id = None
                self.current_crew_name = None
                self.update_current_crew_button()
            self.refresh_crews_list()

    def set_default_crew(self, crew_id):
        with QMutexLocker(self.db_mutex):
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

    def toggle_theme(self):
        self.dark = not self.dark
        self.apply_theme()

    def refresh_crews_list(self):
        self.crew_list.clear()
        crews = self.db.list_crews()
        for crew in crews:
            prefix = "⭐ " if crew['is_default'] else ""
            agents = len(json.loads(crew['config']))
            item_text = f"{prefix}{crew['name']} ({agents} agents)"
            item = QListWidgetItem(item_text)
            roles = " | ".join(a['role'] for a in json.loads(crew['config']))
            item.setToolTip(roles)
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

    def load_crew_template(self):
        names = [t["name"] for t in CREW_TEMPLATES]
        name, ok = QInputDialog.getItem(self, "Load Template", "Choose template:", names, 0, False)
        if ok:
            tmpl = next(t for t in CREW_TEMPLATES if t["name"] == name)
            dialog = CrewConfigDialog(self.models, tmpl["config"], tmpl["name"], self)
            if dialog.exec_() == QDialog.Accepted:
                new_name, config = dialog.get_crew_data()
                if new_name and config:
                    with QMutexLocker(self.db_mutex):
                        nid = self.db.create_crew(new_name, config)
                    self.refresh_crews_list()
                    self.chat.append(f"\n✅ Template '{name}' loaded as '{new_name}'\n")

    def send(self):
        prompt = self.input.toPlainText().strip()
        if not prompt:
            return

        self.last_user_prompt = prompt
        self.input.clear()

        if not self.current_conversation_id:
            title = prompt.split('.')[0][:40] + ("..." if len(prompt.split('.')[0]) > 40 else "")
            with QMutexLocker(self.db_mutex):
                self.current_conversation_id = self.db.create_conversation(title)
            self.refresh_conversations()

        with QMutexLocker(self.db_mutex):
            self.db.add_message(self.current_conversation_id, "user", prompt)
        self.chat.append(f"\n🧑 YOU:\n{prompt}\n")
        if self.attached_image_path:
            cursor = self.chat.textCursor()
            cursor.insertHtml(f'<br><img src="{self.attached_image_path}" width="500"><br><br>')
            self.chat.setTextCursor(cursor)

        mode_text = f" (Crew: {len(self.current_crew_config)} agents)" if self.advanced_mode and self.current_crew_config else ""
        self.chat.append(f"🤖 BOFFIN{mode_text}:\n")
        self.update_stop_reload_button(True)

        max_hist = 20 if self.advanced_mode else 10
        with QMutexLocker(self.db_mutex):
            history = self.db.get_messages(self.current_conversation_id)[-max_hist:]
        ollama_messages = [{"role": m["role"], "content": m["content"]} for m in history]

        if self.attached_image_base64:
            ollama_messages.append({"role": "user", "content": prompt, "images": [self.attached_image_base64]})

        if self.retriever:
            docs = self.retriever.invoke(prompt)
            if docs:
                context = "\n\n".join(d.page_content for d in docs)[:4000]
                rag_block = "Use ONLY these facts if relevant. If unsure, say 'not found':\n\n" + context
                ollama_messages.insert(-1, {"role": "system", "content": rag_block})
                self.chat.append(f"\n🔍 Retrieved {len(docs)} knowledge chunks.\n")

        if self.advanced_mode:
            if not self.current_crew_config:
                self.chat.append("\n❌ No crew selected!\n")
                self.update_stop_reload_button(False)
                return
            crew_cfg = copy.deepcopy(self.current_crew_config)
            if self.retriever:
                crew_cfg[0]['system_prompt'] = (crew_cfg[0].get('system_prompt', '') + "\n\n" + rag_block).strip()
            self.thread = CustomCrewThread(prompt, crew_cfg, history)
        else:
            model = self.model_box.currentText()
            if self.attached_image_base64 and not self.model_vision_cache.get(model, False):
                self.chat.append("\n⚠️ Model does not support vision – image ignored.\n")
                self.attached_image_base64 = None
            self.thread = DirectOllamaThread(model, ollama_messages)

        self.thread.token.connect(self.append_token)
        self.thread.finished.connect(self.on_generation_finished)
        self.thread.error.connect(self.show_error)
        self.thread.start()

        QTimer.singleShot(600000, self.thread.stop)

        self.attached_image_path = None
        self.attached_image_base64 = None

    def append_token(self, text):
        cursor = self.chat.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.chat.setTextCursor(cursor)
        self.chat.ensureCursorVisible()

    def on_generation_finished(self, response, elapsed, chunks):
        if response:
            if hasattr(self.thread, 'is_running') and not self.thread.is_running():
                response += "\n\n[GENERATION STOPPED BY USER]"
            with QMutexLocker(self.db_mutex):
                self.db.add_message(self.current_conversation_id, "assistant", response)
        if chunks:
            speed = len(response) / elapsed if elapsed > 0 else 0
            self.chat.append(f"\n\n📊 {len(response)} chars | {speed:.1f} chars/s | {elapsed:.1f}s\n")
        self.update_stop_reload_button(False)
        if self.thread:
            self.thread.deleteLater()
            self.thread = None

    def show_error(self, e):
        self.chat.append(f"\n❌ Error: {e}\n")
        QMessageBox.critical(self, "Error", str(e))
        self.update_stop_reload_button(False)
        if self.thread:
            self.thread.deleteLater()
            self.thread = None

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    win = OllamaGUI()
    win.show()
    sys.exit(app.exec_())
