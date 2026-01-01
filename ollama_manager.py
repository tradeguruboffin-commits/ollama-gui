#!/usr/bin/env python3
import sys
import subprocess
import threading
import time
import os
import requests
import re
import tempfile
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QListWidget, QLineEdit,
    QMessageBox, QProgressBar, QFileDialog
)

class OllamaManager(QMainWindow):
    # Signals for thread-safe GUI updates
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(bool)
    status_update_signal = pyqtSignal(str, str)  # text, color

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦙 Ollama Manager")
        self.resize(2100, 700)
        self.process = None
        self.server_ready = False
        self.current_modelfile_path = None
        self.current_selected_model = None
        self.is_signed_in = False
        self.init_ui()

        # Connect signals
        self.log_signal.connect(self.append_log)
        self.progress_signal.connect(self.set_progress_visible)
        self.status_update_signal.connect(self.update_auth_status)

        QTimer.singleShot(100, self.check_server_status)

    def append_log(self, text):
        self.log_area.append(text)
        self.log_area.ensureCursorVisible()

    def set_progress_visible(self, visible):
        self.progress.setVisible(visible)

    def update_auth_status(self, text, color):
        self.auth_status_label.setText(text)
        self.auth_status_label.setStyleSheet(f"font-size: 28px; color: {color}; padding: 10px;")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(30)
        layout.setContentsMargins(20, 20, 20, 20)

        left = QVBoxLayout()
        left.setSpacing(30)

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
        self.modelfile_edit.textChanged.connect(self.on_modelfile_changed)
        self.modelfile_edit.setStyleSheet("font-size: 26px;")
        left.addWidget(self.modelfile_edit, 1)

        right = QVBoxLayout()
        right.setSpacing(25)

        right.addWidget(QLabel("<b>📋 Available Models</b>"), alignment=Qt.AlignCenter)

        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self.on_model_select)
        right.addWidget(self.model_list, 1)

        right.addWidget(QLabel("<b>📜 Output Log</b>"), alignment=Qt.AlignCenter)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        right.addWidget(self.log_area, 1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMinimumHeight(50)
        right.addWidget(self.progress)

        del_layout = QHBoxLayout()
        del_layout.setSpacing(20)
        self.selected_label = QLabel("No model selected")
        self.selected_label.setStyleSheet("font-size: 28px;")
        del_layout.addWidget(self.selected_label)

        self.rm_btn = QPushButton("🗑️ Remove")
        self.rm_btn.clicked.connect(self.remove_model)
        self.rm_btn.setEnabled(False)
        self.rm_btn.setStyleSheet("padding: 20px; font-size: 30px;")
        self.rm_btn.setMinimumHeight(70)
        del_layout.addWidget(self.rm_btn)

        self.push_btn = QPushButton("⬆️ Push to ollama.com")
        self.push_btn.clicked.connect(self.push_model)
        self.push_btn.setEnabled(False)
        self.push_btn.setStyleSheet("padding: 20px; font-size: 30px;")
        self.push_btn.setMinimumHeight(70)
        del_layout.addWidget(self.push_btn)

        right.addLayout(del_layout)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(520)
        layout.addWidget(left_widget)

        right_widget = QWidget()
        right_widget.setLayout(right)
        layout.addWidget(right_widget, 1)

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
            QListWidget::item:hover {
                background: #1f6feb50;
            }
            QProgressBar {
                border: 2px solid #30363d; border-radius: 12px; text-align: center;
                background: #161b22; font-size: 26px; min-height: 50px;
            }
            QProgressBar::chunk { background: #238636; }
        """)

        self.log_area.setStyleSheet("""
            background: #0d1117; color: #58a6ff;
            font-family: Consolas, Monaco, monospace; font-size: 26px; padding: 20px;
        """)

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

    def update_push_button(self):
        if not hasattr(self, 'push_btn') or not self.push_btn:
            return
        has_model = bool(hasattr(self, 'current_selected_model') and self.current_selected_model)
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
            self.status_update_signal.emit(f"🟢 Signed in as {username}", "#50fa7b")
            self.signin_btn.setEnabled(False)
            self.signout_btn.setEnabled(True)
            self.is_signed_in = True
        else:
            self.log_signal.emit("🔴 Not signed in.")
            self.status_update_signal.emit("🔴 Not signed in to ollama.com", "#ff5555")
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
            self.status_update_signal.emit(f"🟢 Signed in as {username}", "#50fa7b")
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

            self.status_update_signal.emit("🟡 Pending: Complete in browser", "#fbbc05")
            self.signin_btn.setEnabled(False)
            self.signout_btn.setEnabled(True)
            self.is_signed_in = False

            if hasattr(self, 'auth_poll_timer'):
                self.auth_poll_timer.stop()
            self.auth_poll_timer = QTimer()
            self.auth_poll_timer.timeout.connect(self.check_auth_status)
            self.auth_poll_timer.start(3000)
        else:
            self.log_signal.emit("❌ Unexpected response from signin.")
            self.log_signal.emit(f"   Output: {result.stdout.strip()}")
            self.log_signal.emit(f"   Error: {result.stderr.strip()}")

    def check_auth_status(self):
        result = subprocess.run(["ollama", "signin"], capture_output=True, text=True)
        full = result.stdout + result.stderr
        match = re.search(r"already signed in as user ['\"]?([a-zA-Z0-9_]+)['\"]?", full, re.IGNORECASE)
        if match:
            username = match.group(1)
            self.log_signal.emit("✅ Authentication completed!")
            self.status_update_signal.emit(f"🟢 Signed in as {username}", "#50fa7b")
            self.is_signed_in = True
            self.update_push_button()
            self.auth_poll_timer.stop()

    def signout(self):
        reply = QMessageBox.question(self, "Confirm", "Sign out from ollama.com?")
        if reply == QMessageBox.Yes:
            result = subprocess.run(["ollama", "signout"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_signal.emit("🚪 Signed out successfully.")
                self.status_update_signal.emit("🔴 Not signed in to ollama.com", "#ff5555")
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
            QMessageBox.warning(self, "Warning", "Model name must include username (e.g. username/my-model)")
            return

        self.progress_signal.emit(True)
        self.log_signal.emit(f"⬆️ Pushing {model}...")

        def push():
            try:
                proc = subprocess.Popen(["ollama", "push", model], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.log_signal.emit(line.strip())
                proc.wait()
                if proc.returncode == 0:
                    self.log_signal.emit(f"✅ {model} pushed successfully!")
                else:
                    self.log_signal.emit("❌ Push failed")
            except Exception as e:
                self.log_signal.emit(f"❌ Error: {e}")
            finally:
                self.progress_signal.emit(False)

        threading.Thread(target=push, daemon=True).start()

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

    def pull_model(self):
        model = self.pull_input.text().strip()
        if not model:
            QMessageBox.warning(self, "Error", "Enter a model name!")
            return
        if not self.server_ready:
            QMessageBox.warning(self, "Error", "Start server first!")
            return

        self.progress_signal.emit(True)
        self.log_signal.emit(f"⬇️ Pulling {model}...")

        def pull():
            try:
                proc = subprocess.Popen(["ollama", "pull", model], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.log_signal.emit(line.strip())
                proc.wait()
                self.log_signal.emit(f"✅ {model} pulled successfully!")
                self.load_models()
            except Exception as e:
                self.log_signal.emit(f"❌ Error: {e}")
            finally:
                self.progress_signal.emit(False)

        threading.Thread(target=pull, daemon=True).start()

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
            except Exception as e:
                self.log_signal.emit(f"❌ {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OllamaManager()
    win.show()
    sys.exit(app.exec_())
