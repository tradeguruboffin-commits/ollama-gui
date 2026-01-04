#!/usr/bin/env python3
import sys
import subprocess
import threading
import time
import os
import requests
import re
import tempfile
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QListWidget, QLineEdit,
    QMessageBox, QProgressBar, QFileDialog, QFrame
)

class OllamaManager(QMainWindow):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)  # 0-100
    progress_visible_signal = pyqtSignal(bool)
    auth_status_signal = pyqtSignal(str, str)  # text, color

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦙 Ollama Manager")
        self.resize(2500, 900)
        self.process = None
        self.server_ready = False
        self.current_modelfile_path = None
        self.current_selected_model = None
        self.is_signed_in = False
        self.init_ui()

        self.log_signal.connect(self.append_log)
        self.progress_signal.connect(self.progress.setValue)
        self.progress_visible_signal.connect(self.progress.setVisible)
        self.auth_status_signal.connect(self.update_auth_label)

        QTimer.singleShot(100, self.check_server_status)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left Panel
        left = QVBoxLayout()
        left.setSpacing(25)

        self.status_label = QLabel("🔴 Ollama Server: Checking...")
        self.status_label.setStyleSheet("font-size: 32px; font-weight: bold; padding: 20px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.status_label)

        self.serve_btn = QPushButton("▶️ Start Ollama Serve")
        self.serve_btn.clicked.connect(self.toggle_serve)
        self.serve_btn.setStyleSheet("padding: 20px; font-size: 30px;")
        self.serve_btn.setMinimumHeight(80)
        left.addWidget(self.serve_btn)

        self.refresh_btn = QPushButton("🔄 Refresh Model List")
        self.refresh_btn.clicked.connect(self.load_models)
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setStyleSheet("padding: 18px; font-size: 28px;")
        self.refresh_btn.setMinimumHeight(70)
        left.addWidget(self.refresh_btn)

        auth_layout = QHBoxLayout()
        auth_layout.setSpacing(15)
        self.auth_status_label = QLabel("🔴 Not signed in to ollama.com")
        self.auth_status_label.setStyleSheet("font-size: 28px; color: #ff5555; padding: 10px;")
        auth_layout.addWidget(self.auth_status_label)

        self.signin_btn = QPushButton("🔑 Sign In")
        self.signin_btn.clicked.connect(self.signin)
        self.signin_btn.setStyleSheet("padding: 18px; font-size: 28px;")
        self.signin_btn.setMinimumHeight(70)
        auth_layout.addWidget(self.signin_btn, 1)

        self.signout_btn = QPushButton("🚪 Sign Out")
        self.signout_btn.clicked.connect(self.signout)
        self.signout_btn.setEnabled(False)
        self.signout_btn.setStyleSheet("padding: 18px; font-size: 28px;")
        self.signout_btn.setMinimumHeight(70)
        auth_layout.addWidget(self.signout_btn, 1)

        left.addLayout(auth_layout)

        pull_layout = QHBoxLayout()
        pull_layout.setSpacing(15)
        self.pull_input = QLineEdit()
        self.pull_input.setPlaceholderText("e.g. llama3.2, qwen2.5:7b, gemma2...")
        self.pull_input.setStyleSheet("font-size: 28px; padding: 15px;")
        self.pull_input.setMinimumHeight(70)
        pull_layout.addWidget(self.pull_input, 3)
        self.pull_btn = QPushButton("⬇️ Pull")
        self.pull_btn.clicked.connect(self.pull_model)
        self.pull_btn.setStyleSheet("padding: 20px; font-size: 30px;")
        self.pull_btn.setMinimumHeight(70)
        pull_layout.addWidget(self.pull_btn, 1)
        left.addLayout(pull_layout)

        create_name_layout = QHBoxLayout()
        create_name_layout.setSpacing(15)
        self.create_name = QLineEdit()
        self.create_name.setPlaceholderText("Custom model name (e.g. username/my-llama3)")
        self.create_name.setStyleSheet("font-size: 28px; padding: 15px;")
        self.create_name.setMinimumHeight(70)
        self.create_name.textChanged.connect(self.check_create_button)  # <-- নতুন যোগ
        create_name_layout.addWidget(self.create_name, 3)

        self.create_btn = QPushButton("🛠️ Create Model")
        self.create_btn.clicked.connect(self.create_model)
        self.create_btn.setEnabled(False)
        self.create_btn.setStyleSheet("padding: 20px; font-size: 30px;")
        self.create_btn.setMinimumHeight(70)
        create_name_layout.addWidget(self.create_btn, 1)
        left.addLayout(create_name_layout)

        modelfile_btn_layout = QHBoxLayout()
        modelfile_btn_layout.setSpacing(15)
        self.modelfile_path_label = QLabel("No Modelfile selected")
        self.modelfile_path_label.setStyleSheet("font-size: 26px; color: #8b949e;")
        modelfile_btn_layout.addWidget(self.modelfile_path_label)

        self.browse_modelfile_btn = QPushButton("📂 Browse Modelfile")
        self.browse_modelfile_btn.clicked.connect(self.browse_modelfile)
        self.browse_modelfile_btn.setStyleSheet("padding: 18px; font-size: 28px;")
        self.browse_modelfile_btn.setMinimumHeight(70)
        modelfile_btn_layout.addWidget(self.browse_modelfile_btn)
        left.addLayout(modelfile_btn_layout)

        left.addWidget(QLabel("📄 Modelfile Content:"), alignment=Qt.AlignCenter)
        self.modelfile_edit = QTextEdit()
        self.modelfile_edit.setPlaceholderText("""# Example Modelfile
FROM llama3.2
PARAMETER temperature 0.8
SYSTEM You are a helpful assistant.""")
        self.modelfile_edit.textChanged.connect(self.check_create_button)
        self.modelfile_edit.setStyleSheet("font-size: 26px;")
        left.addWidget(self.modelfile_edit, 1)

        # Right Panel - Models List + Details
        right = QVBoxLayout()
        right.setSpacing(20)

        right.addWidget(QLabel("<b>📋 Available Models</b>"), alignment=Qt.AlignCenter)

        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self.on_model_select)
        right.addWidget(self.model_list, 2)

        # Model Details Panel
        details_frame = QFrame()
        details_frame.setFrameShape(QFrame.StyledPanel)
        details_frame.setStyleSheet("background: #161b22; border: 2px solid #30363d; border-radius: 15px; padding: 15px;")
        details_layout = QVBoxLayout(details_frame)
        details_layout.addWidget(QLabel("<b>📄 Model Details</b>"), alignment=Qt.AlignCenter)
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet("background: #0d1117; color: #c9d1d9; font-size: 24px;")
        self.details_text.setMinimumHeight(200)
        self.details_text.setPlaceholderText("Select a model to see details...")
        details_layout.addWidget(self.details_text)
        right.addWidget(details_frame, 1)

        right.addWidget(QLabel("<b>📜 Output Log</b>"), alignment=Qt.AlignCenter)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        right.addWidget(self.log_area, 2)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet("font-size: 26px; min-height: 60px;")
        self.progress.setVisible(False)
        right.addWidget(self.progress)

        # Bottom buttons on right
        bottom_right = QHBoxLayout()
        bottom_right.setSpacing(20)
        self.selected_label = QLabel("No model selected")
        self.selected_label.setStyleSheet("font-size: 28px;")
        bottom_right.addWidget(self.selected_label)

        self.rm_btn = QPushButton("🗑️ Remove")
        self.rm_btn.clicked.connect(self.remove_model)
        self.rm_btn.setEnabled(False)
        self.rm_btn.setStyleSheet("padding: 20px; font-size: 30px;")
        self.rm_btn.setMinimumHeight(70)
        bottom_right.addWidget(self.rm_btn)

        self.push_btn = QPushButton("⬆️ Push to ollama.com")
        self.push_btn.clicked.connect(self.push_model)
        self.push_btn.setEnabled(False)
        self.push_btn.setStyleSheet("padding: 20px; font-size: 30px;")
        self.push_btn.setMinimumHeight(70)
        bottom_right.addWidget(self.push_btn)

        right.addLayout(bottom_right)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(550)
        main_layout.addWidget(left_widget)

        right_widget = QWidget()
        right_widget.setLayout(right)
        main_layout.addWidget(right_widget, 1)

        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #0d1117; color: #c9d1d9; }
            QLabel { color: #c9d1d9; font-size: 30px; font-weight: bold; }
            QPushButton {
                background: #21262d; color: #c9d1d9; border: 2px solid #30363d;
                padding: 20px; border-radius: 15px; font-size: 30px;
            }
            QPushButton:hover { background: #30363d; }
            QPushButton:pressed { background: #444c56; }
            QPushButton:disabled { background: #161b22; color: #6e7681; }
            QLineEdit, QTextEdit {
                background: #161b22; color: #c9d1d9; border: 2px solid #30363d;
                padding: 20px; border-radius: 12px; font-size: 28px;
            }
            QListWidget {
                background: #161b22; color: #c9d1d9; border: 2px solid #30363d;
                border-radius: 15px; font-size: 28px; padding: 15px;
            }
            QListWidget::item {
                padding: 25px 15px; min-height: 70px; border-bottom: 2px solid #21262d;
            }
            QListWidget::item:selected {
                background: #264f78; color: white; font-weight: bold;
            }
            QProgressBar {
                border: 2px solid #30363d; border-radius: 12px; text-align: center;
                background: #161b22; font-size: 28px; min-height: 60px;
            }
            QProgressBar::chunk { background: #238636; }
        """)

        self.log_area.setStyleSheet("""
            background: #0d1117; color: #58a6ff;
            font-family: Consolas, Monaco, monospace; font-size: 26px; padding: 20px;
        """)

    def append_log(self, text):
        self.log_area.append(text)
        self.log_area.ensureCursorVisible()

    def update_auth_label(self, text, color):
        self.auth_status_label.setText(text)
        self.auth_status_label.setStyleSheet(f"font-size: 28px; color: {color}; padding: 10px;")

    def is_server_running(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=4)
            return response.status_code == 200
        except:
            return False

    def check_server_status(self):
        if self.is_server_running():
            self.server_ready = True
            self.status_label.setText("🟢 Ollama Server: Running")
            self.serve_btn.setText("⏹️ Stop Ollama Serve")
            self.refresh_btn.setEnabled(True)
            self.load_models()
        else:
            self.server_ready = False
            self.status_label.setText("🔴 Ollama Server: Stopped")
            self.serve_btn.setText("▶️ Start Ollama Serve")
            self.refresh_btn.setEnabled(False)
            self.model_list.clear()
        self.update_push_button()

        QTimer.singleShot(500, self.check_initial_auth_status)

    def toggle_serve(self):
        if self.server_ready:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()  # <-- Force kill if terminate fails
                    self.log_signal.emit("🔴 Force killed ollama serve")
                self.process = None
            self.log_signal.emit("🛑 Ollama serve stopped.")
            self.check_server_status()
        else:
            thread = threading.Thread(target=self.run_serve_background)
            thread.daemon = True
            thread.start()

    def run_serve_background(self):
        self.log_signal.emit("▶️ Starting ollama serve...")
        self.process = subprocess.Popen(["ollama", "serve"])
        for _ in range(30):
            time.sleep(1)
            if self.is_server_running():
                self.check_server_status()
                self.log_signal.emit("🟢 Ollama server ready!")
                return
        self.log_signal.emit("⚠️ Server started but not responding. Try Refresh.")

    def load_models(self):
        if not self.is_server_running():
            self.log_signal.emit("⚠️ Server not responding.")
            return

        self.model_list.clear()
        try:
            response = requests.get("http://localhost:11434/api/tags")
            data = response.json()
            models = data.get("models", [])

            for model in models:
                name = model["name"]
                size_bytes = model.get("size", 0)
                modified = model.get("modified_at", "Unknown")
                if modified != "Unknown":
                    modified = modified.split("T")[0]

                size_str = self.format_size(size_bytes)
                item_text = f"{name}  |  Size: {size_str}  |  Modified: {modified}"
                self.model_list.addItem(item_text)

            self.log_signal.emit(f"✅ Loaded {len(models)} models.")
        except Exception as e:
            self.log_signal.emit(f"❌ Failed to load models: {str(e)}")

    def format_size(self, size_bytes):
        if size_bytes == 0 or size_bytes is None:
            return "Unknown"
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        size = float(size_bytes)
        i = 0
        while size >= 1024.0 and i < len(units)-1:
            size /= 1024.0
            i += 1
        if size >= 100:
            return f"{size:.0f} {units[i]}"
        elif size >= 10:
            return f"{size:.1f} {units[i]}"
        else:
            return f"{size:.2f} {units[i]}"

    def on_model_select(self, item):
        model_name = item.text().split("  |  ")[0]
        self.selected_label.setText(f"Selected: {model_name}")
        self.rm_btn.setEnabled(True)
        self.current_selected_model = model_name
        self.update_push_button()
        self.show_model_details(model_name)

    def show_model_details(self, model_name):
        try:
            response = requests.post("http://localhost:11434/api/show", json={"name": model_name})
            if response.status_code == 200:
                info = response.json()
                details = f"<b>Model:</b> {info.get('modelfile', '').split('FROM ')[1].split('\n')[0] if 'FROM ' in info.get('modelfile', '') else model_name}<br>"
                details += f"<b>Parameters:</b> {info.get('parameters', 'Unknown')}<br>"
                details += f"<b>Template:</b> {info.get('template', 'Unknown')[:100]}...<br>"
                details += f"<b>Digest:</b> {info.get('digest', 'Unknown')[:20]}...<br>"
                details += f"<b>Size:</b> {self.format_size(info.get('size', 0))}<br>"
                details += f"<b>Quantization:</b> {info.get('quantization', 'Unknown')}"
                self.details_text.setHtml(details)
            else:
                self.details_text.setText("Failed to load details")
        except:
            self.details_text.setText("Server not responding")

    def update_push_button(self):
        if not hasattr(self, 'push_btn'):
            return
        has_model = bool(self.current_selected_model)
        has_username = has_model and '/' in self.current_selected_model
        enabled = has_username and self.is_signed_in
        self.push_btn.setEnabled(enabled)

    def check_initial_auth_status(self):
        self.log_signal.emit("🔍 Checking initial authentication status...")
        result = subprocess.run(["ollama", "signin"], capture_output=True, text=True)
        full_output = result.stdout + result.stderr
        match = re.search(r"already signed in as user ['\"]?([a-zA-Z0-9_]+)['\"]?", full_output, re.IGNORECASE)
        if match:
            username = match.group(1)
            self.log_signal.emit(f"✅ Already signed in as: {username}")
            self.auth_status_signal.emit(f"🟢 Signed in as {username}", "#50fa7b")
            self.signin_btn.setEnabled(False)
            self.signout_btn.setEnabled(True)
            self.is_signed_in = True
        else:
            self.log_signal.emit("🔴 Not signed in.")
            self.auth_status_signal.emit("🔴 Not signed in to ollama.com", "#ff5555")
            self.is_signed_in = False
        self.update_push_button()

    def signin(self):
        self.log_signal.emit("🔑 Checking sign-in status...")
        result = subprocess.run(["ollama", "signin"], capture_output=True, text=True)
        full_output = result.stdout + result.stderr

        match = re.search(r"already signed in as user ['\"]?([a-zA-Z0-9_]+)['\"]?", full_output, re.IGNORECASE)
        if match:
            username = match.group(1)
            self.log_signal.emit(f"✅ Already signed in as: {username}")
            self.auth_status_signal.emit(f"🟢 Signed in as {username}", "#50fa7b")
            self.signin_btn.setEnabled(False)
            self.signout_btn.setEnabled(True)
            self.is_signed_in = True
            self.update_push_button()
            return

        url_match = re.search(r"https?://[^\s]+", full_output)
        if url_match:
            url = url_match.group(0).strip()
            self.log_signal.emit("   📎 Authentication required:")
            self.log_signal.emit(f"   {url}")
            self.log_signal.emit("   Opening browser...")
            try:
                subprocess.run(["xdg-open", url], check=False)
                self.log_signal.emit("   🌐 Browser opened.")
            except:
                self.log_signal.emit("   ⚠️ Auto-open failed. Copy URL manually.")

            self.auth_status_signal.emit("🟡 Pending: Complete in browser", "#fbbc05")
            self.signin_btn.setEnabled(False)
            self.signout_btn.setEnabled(True)
            self.is_signed_in = False

            if hasattr(self, 'auth_poll_timer'):
                self.auth_poll_timer.stop()
            self.auth_poll_timer = QTimer()
            self.auth_poll_timer.timeout.connect(self.check_auth_status)
            self.auth_poll_timer.start(3000)

    def check_auth_status(self):
        result = subprocess.run(["ollama", "signin"], capture_output=True, text=True)
        full = result.stdout + result.stderr
        match = re.search(r"already signed in as user ['\"]?([a-zA-Z0-9_]+)['\"]?", full, re.IGNORECASE)
        if match:
            username = match.group(1)
            self.log_signal.emit("✅ Authentication completed!")
            self.auth_status_signal.emit(f"🟢 Signed in as {username}", "#50fa7b")
            self.is_signed_in = True
            self.update_push_button()
            self.auth_poll_timer.stop()

    def signout(self):
        reply = QMessageBox.question(self, "Confirm", "Sign out from ollama.com?")
        if reply == QMessageBox.Yes:
            result = subprocess.run(["ollama", "signout"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_signal.emit("🚪 Signed out successfully.")
                self.auth_status_signal.emit("🔴 Not signed in to ollama.com", "#ff5555")
                self.signin_btn.setEnabled(True)
                self.signout_btn.setEnabled(False)
                self.is_signed_in = False
                self.update_push_button()
                if hasattr(self, 'auth_poll_timer'):
                    self.auth_poll_timer.stop()
            else:
                self.log_signal.emit(f"❌ Sign out failed: {result.stderr}")

    def push_model(self):
        model = self.current_selected_model
        if '/' not in model:
            QMessageBox.warning(self, "Warning", "Model name must include username")
            return

        self.progress_visible_signal.emit(True)
        self.progress_signal.emit(0)
        self.log_signal.emit(f"⬆️ Pushing {model}...")

        def push():
            try:
                proc = subprocess.Popen(["ollama", "push", model], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    line = line.strip()
                    if line:
                        self.log_signal.emit(line)
                        # Parse percentage
                        m = re.search(r"(\d+)%", line)
                        if m:
                            self.progress_signal.emit(int(m.group(1)))
                returncode = proc.wait()
                if returncode == 0:
                    self.log_signal.emit(f"✅ {model} pushed successfully!")
                else:
                    self.log_signal.emit(f"❌ Push failed (code: {returncode})")
            except Exception as e:
                self.log_signal.emit(f"❌ Error: {e}")
            finally:
                self.progress_visible_signal.emit(False)

        threading.Thread(target=push, daemon=True).start()

    def pull_model(self):
        model = self.pull_input.text().strip()
        if not model:
            QMessageBox.warning(self, "Error", "Enter a model name!")
            return
        if not self.server_ready:
            QMessageBox.warning(self, "Error", "Start server first!")
            return

        self.progress_visible_signal.emit(True)
        self.progress_signal.emit(0)
        self.log_signal.emit(f"⬇️ Pulling {model}...")

        def pull():
            try:
                proc = subprocess.Popen(["ollama", "pull", model], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    line = line.strip()
                    if line:
                        self.log_signal.emit(line)
                        m = re.search(r"(\d+)%", line)
                        if m:
                            self.progress_signal.emit(int(m.group(1)))
                returncode = proc.wait()
                if returncode == 0:
                    self.log_signal.emit(f"✅ {model} pulled successfully!")
                    self.load_models()
                else:
                    self.log_signal.emit(f"❌ Pull failed (code: {returncode})")
            except Exception as e:
                self.log_signal.emit(f"❌ Error: {e}")
            finally:
                self.progress_visible_signal.emit(False)

        threading.Thread(target=pull, daemon=True).start()

    def browse_modelfile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Modelfile", "", "Modelfile (*);;All Files (*)")
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.modelfile_edit.setPlainText(content)
                self.modelfile_path_label.setText(os.path.basename(path))
                self.log_signal.emit(f"📂 Loaded: {os.path.basename(path)}")
                self.check_create_button()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Read error:\n{str(e)}")

    def on_modelfile_changed(self):
        self.check_create_button()

    def check_create_button(self):
        name = self.create_name.text().strip()
        content = self.modelfile_edit.toPlainText().strip()
        self.create_btn.setEnabled(bool(name and content))

    def create_model(self):
        name = self.create_name.text().strip()
        content = self.modelfile_edit.toPlainText().strip()
        if not name or not content:
            return
        if not self.server_ready:
            QMessageBox.warning(self, "Error", "Start server first!")
            return

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=True) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            self.log_signal.emit(f"🛠️ Creating {name}...")
            result = subprocess.run(["ollama", "create", name, "-f", temp_file.name], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_signal.emit(f"✅ '{name}' created!")
                self.load_models()
            else:
                self.log_signal.emit(f"❌ Failed:\n{result.stderr}")
                QMessageBox.critical(self, "Error", result.stderr or "Unknown")

    def remove_model(self):
        model = self.current_selected_model
        reply = QMessageBox.question(self, "Confirm", f"Permanently delete '{model}'?")
        if reply == QMessageBox.Yes:
            try:
                subprocess.run(["ollama", "rm", model], check=True)
                self.log_signal.emit(f"🗑️ Removed {model}")
                self.load_models()
                self.selected_label.setText("No model selected")
                self.rm_btn.setEnabled(False)
                self.push_btn.setEnabled(False)
                self.details_text.setText("Select a model to see details...")
            except Exception as e:
                self.log_signal.emit(f"❌ {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OllamaManager()
    win.show()
    sys.exit(app.exec_())
