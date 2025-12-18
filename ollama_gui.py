#!/usr/bin/env python3
# JARVIS CORE – ALL IN (STABLE FIXED BUILD)

import sys, os, json, threading, subprocess, requests
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor

# ---------- CONFIG ----------

OLLAMA_URL = "http://localhost:11434/api/chat"
CONFIG_FILE = "jarvis_config.json"
MEMORY_FILE = "jarvis_memory.json"

def load_json(p, d):
    if os.path.exists(p):
        try: return json.load(open(p))
        except: pass
    return d

def save_json(p, d):
    json.dump(d, open(p,"w"), indent=2)

config = load_json(CONFIG_FILE, {"theme":"dark","model":"phi3:mini"})
memory = load_json(MEMORY_FILE, [])

# ---------- MODELS ----------

def get_models():
    try:
        out = subprocess.check_output(["ollama","list"], text=True)
        return [l.split()[0] for l in out.splitlines()[1:] if l.strip()]
    except:
        return [config["model"]]

# ---------- SIGNAL BRIDGE ----------

class StreamBridge(QObject):
    token = pyqtSignal(str)
    done = pyqtSignal(str)
    error = pyqtSignal(str)

bridge = StreamBridge()

# ---------- UI ----------

class Jarvis(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JARVIS CORE – ALL IN")
        self.resize(1200,720)
        self.setFont(QFont("Consolas",11))
        self.setStyleSheet(self.dark() if config["theme"]=="dark" else self.light())
        self.init_ui()

        bridge.token.connect(self.append_token)
        bridge.done.connect(self.finish_reply)
        bridge.error.connect(lambda e: self.log("Error", e))

    def init_ui(self):
        c = QWidget(); self.setCentralWidget(c)
        v = QVBoxLayout(c)

        # Top bar
        t = QHBoxLayout()
        self.status = QLabel("🟢 JARVIS READY")
        self.model = QComboBox()
        self.model.addItems(get_models())
        self.model.setCurrentText(config["model"])
        self.model.currentTextChanged.connect(self.set_model)

        self.theme_btn = QPushButton("Theme")
        self.theme_btn.clicked.connect(self.toggle_theme)

        t.addWidget(self.status)
        t.addStretch()
        t.addWidget(self.theme_btn)
        t.addWidget(self.model)
        v.addLayout(t)

        # Chat
        self.chat = QTextEdit(readOnly=True)
        v.addWidget(self.chat,1)

        # Bottom
        b = QHBoxLayout()
        self.input = QTextEdit(); self.input.setFixedHeight(90)

        send = QPushButton("Send"); send.clicked.connect(self.send)
        clear = QPushButton("Clear"); clear.clicked.connect(self.clear_chat)
        export = QPushButton("Export"); export.clicked.connect(self.export_chat)

        b.addWidget(self.input,1)
        b.addWidget(send); b.addWidget(clear); b.addWidget(export)
        v.addLayout(b)

        self.log("System","JARVIS ALL-IN READY")

    # ---------- THEMES ----------

    def dark(self):
        return """
        QWidget{background:#000;color:#00ff88;}
        QTextEdit{background:#000;border:1px solid #00ff88;}
        QPushButton{background:#111;border:1px solid #00ff88;padding:6px;}
        QPushButton:hover{background:#003322;}
        """

    def light(self):
        return "QWidget{background:#fff;color:#000;}"

    def toggle_theme(self):
        config["theme"] = "light" if config["theme"]=="dark" else "dark"
        save_json(CONFIG_FILE, config)
        self.setStyleSheet(self.dark() if config["theme"]=="dark" else self.light())

    # ---------- CHAT ----------

    def log(self, who, text):
        self.chat.append(f"<b>{who} ▶</b> {text}")

    def append_token(self, t):
        self.chat.moveCursor(QTextCursor.End)
        self.chat.insertPlainText(t)

    def finish_reply(self, full):
        memory.append({"role":"assistant","content":full})
        save_json(MEMORY_FILE, memory)
        self.chat.append("")

    def set_model(self, m):
        config["model"] = m
        save_json(CONFIG_FILE, config)

    def clear_chat(self):
        self.chat.clear()
        memory.clear()
        save_json(MEMORY_FILE, memory)

    def export_chat(self):
        p,_ = QFileDialog.getSaveFileName(self,"Export","chat.txt")
        if p: open(p,"w").write(self.chat.toPlainText())

    # ---------- STREAM ----------

    def send(self):
        msg = self.input.toPlainText().strip()
        if not msg: return
        self.input.clear()
        self.log("YOU", msg)
        self.chat.append("<b>JARVIS ▶</b> ")
        memory.append({"role":"user","content":msg})
        save_json(MEMORY_FILE, memory)
        threading.Thread(target=ollama_stream, args=(memory[-10:],), daemon=True).start()

# ---------- STREAM WORKER ----------

def ollama_stream(msgs):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model":config["model"],
            "messages":msgs,
            "stream":True
        }, stream=True)

        full=""
        for line in r.iter_lines():
            if not line: continue
            j = json.loads(line.decode())
            if "message" in j:
                tok = j["message"]["content"]
                full += tok
                bridge.token.emit(tok)

        bridge.done.emit(full)

    except Exception as e:
        bridge.error.emit(str(e))

# ---------- RUN ----------

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    win = Jarvis()
    win.show()
    sys.exit(app.exec_())
