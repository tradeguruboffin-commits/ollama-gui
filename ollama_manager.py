#!/usr/bin/env python3
import sys
import subprocess
import threading
import time
import os
import requests
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QListWidget, QLineEdit,
    QMessageBox, QProgressBar, QFileDialog
)

class OllamaManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦙 Ollama Model Manager")
        self.resize(1200, 780)
        self.process = None
        self.server_ready = False
        self.current_modelfile_path = None
        self.init_ui()
        self.check_server_status()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left Panel
        left = QVBoxLayout()
        left.setSpacing(20)

        self.status_label = QLabel("🔴 Ollama Server: Checking...")
        self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 12px;")
        left.addWidget(self.status_label)

        self.serve_btn = QPushButton("▶️ Start Ollama Serve")
        self.serve_btn.clicked.connect(self.toggle_serve)
        self.serve_btn.setStyleSheet("padding: 16px; font-size: 18px;")
        left.addWidget(self.serve_btn)

        self.refresh_btn = QPushButton("🔄 Refresh Model List")
        self.refresh_btn.clicked.connect(self.load_models)
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setStyleSheet("padding: 14px; font-size: 16px;")
        left.addWidget(self.refresh_btn)

        # Pull Model
        pull_layout = QHBoxLayout()
        self.pull_input = QLineEdit()
        self.pull_input.setPlaceholderText("e.g. llama3.2, qwen2.5:7b, gemma2...")
        self.pull_input.setStyleSheet("font-size: 16px; padding: 10px;")
        pull_layout.addWidget(self.pull_input)
        self.pull_btn = QPushButton("⬇️ Pull")
        self.pull_btn.clicked.connect(self.pull_model)
        self.pull_btn.setStyleSheet("padding: 12px; font-size: 16px;")
        pull_layout.addWidget(self.pull_btn)
        left.addLayout(pull_layout)

        # Create Custom Model
        create_name_layout = QHBoxLayout()
        self.create_name = QLineEdit()
        self.create_name.setPlaceholderText("Custom model name (e.g. my-llama3)")
        self.create_name.setStyleSheet("font-size: 16px; padding: 10px;")
        create_name_layout.addWidget(self.create_name)

        self.create_btn = QPushButton("🛠️ Create Model")
        self.create_btn.clicked.connect(self.create_model)
        self.create_btn.setEnabled(False)
        self.create_btn.setStyleSheet("padding: 14px; font-size: 18px;")
        create_name_layout.addWidget(self.create_btn)
        left.addLayout(create_name_layout)

        # Modelfile Browse Button
        modelfile_btn_layout = QHBoxLayout()
        self.modelfile_path_label = QLabel("No Modelfile selected")
        self.modelfile_path_label.setStyleSheet("font-size: 14px; color: #8b949e;")
        modelfile_btn_layout.addWidget(self.modelfile_path_label)

        self.browse_modelfile_btn = QPushButton("📂 Browse Modelfile")
        self.browse_modelfile_btn.clicked.connect(self.browse_modelfile)
        self.browse_modelfile_btn.setStyleSheet("padding: 12px; font-size: 16px;")
        modelfile_btn_layout.addWidget(self.browse_modelfile_btn)
        left.addLayout(modelfile_btn_layout)

        # Modelfile Editor
        left.addWidget(QLabel("📄 Modelfile Content:"))
        self.modelfile_edit = QTextEdit()
        self.modelfile_edit.setPlaceholderText("""# Example Modelfile
FROM llama3.2
PARAMETER temperature 0.8
SYSTEM You are a helpful assistant.""")
        self.modelfile_edit.textChanged.connect(self.on_modelfile_changed)
        self.modelfile_edit.setStyleSheet("font-size: 16px;")
        left.addWidget(self.modelfile_edit)

        left.addStretch()

        # Right Panel
        right = QVBoxLayout()
        right.setSpacing(15)

        right.addWidget(QLabel("<b>📋 Available Models</b>"), alignment=Qt.AlignCenter)

        self.model_list = QListWidget()
        self.model_list.itemClicked.connect(self.on_model_select)
        right.addWidget(self.model_list)

        right.addWidget(QLabel("<b>📜 Output Log</b>"), alignment=Qt.AlignCenter)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        right.addWidget(self.log_area)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        right.addWidget(self.progress)

        del_layout = QHBoxLayout()
        self.selected_label = QLabel("No model selected")
        self.selected_label.setStyleSheet("font-size: 16px;")
        del_layout.addWidget(self.selected_label)
        self.rm_btn = QPushButton("🗑️ Remove")
        self.rm_btn.clicked.connect(self.remove_model)
        self.rm_btn.setEnabled(False)
        self.rm_btn.setStyleSheet("padding: 12px; font-size: 16px;")
        del_layout.addWidget(self.rm_btn)
        right.addLayout(del_layout)

        # Assemble
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(460)
        layout.addWidget(left_widget)

        right_widget = QWidget()
        right_widget.setLayout(right)
        layout.addWidget(right_widget, 1)

        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #0d1117; color: #c9d1d9; }
            QLabel { color: #c9d1d9; font-size: 16px; }
            QPushButton {
                background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
                padding: 12px; border-radius: 10px; font-size: 16px;
            }
            QPushButton:hover { background: #30363d; }
            QPushButton:pressed { background: #444c56; }
            QPushButton:disabled { background: #161b22; color: #6e7681; }
            QLineEdit, QTextEdit {
                background: #161b22; color: #c9d1d9; border: 1px solid #30363d;
                padding: 12px; border-radius: 8px; font-size: 16px;
            }
            QListWidget {
                background: #161b22; color: #c9d1d9; border: 1px solid #30363d;
                border-radius: 10px; font-size: 16px; padding: 8px;
            }
            QListWidget::item {
                padding: 16px 10px; min-height: 40px; border-bottom: 1px solid #21262d;
            }
            QListWidget::item:selected {
                background: #264f78; color: white; font-weight: bold;
            }
            QListWidget::item:hover {
                background: #1f6feb40;
            }
            QProgressBar {
                border: 1px solid #30363d; border-radius: 8px; text-align: center;
                background: #161b22; font-size: 14px;
            }
            QProgressBar::chunk { background: #238636; }
        """)

        self.log_area.setStyleSheet("""
            background: #0d1117; color: #58a6ff;
            font-family: Consolas, Monaco, monospace; font-size: 15px;
        """)

    def log(self, text):
        self.log_area.append(text)
        self.log_area.ensureCursorVisible()

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

    def toggle_serve(self):
        if self.server_ready:
            if self.process:
                self.process.terminate()
                self.process = None
            self.log("🛑 Ollama serve stopped.")
            self.check_server_status()
        else:
            def run_serve():
                self.log("▶️ Starting ollama serve...")
                self.process = subprocess.Popen(["ollama", "serve"])
                for _ in range(30):
                    time.sleep(1)
                    if self.is_server_running():
                        self.check_server_status()
                        self.log("🟢 Ollama server ready!")
                        return
                self.log("⚠️ Server started but not responding. Try Refresh.")

            threading.Thread(target=run_serve, daemon=True).start()

    def load_models(self):
        if not self.is_server_running():
            self.log("⚠️ Server not responding.")
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

            self.log(f"✅ Loaded {len(models)} models.")
        except Exception as e:
            self.log(f"❌ Failed: {str(e)}")

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

    def browse_modelfile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Modelfile", "", "Modelfile (*);;Text Files (*.txt);;All Files (*)"
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.modelfile_edit.setPlainText(content)
                self.current_modelfile_path = path
                self.modelfile_path_label.setText(os.path.basename(path))
                self.log(f"📂 Loaded Modelfile: {os.path.basename(path)}")
                self.check_create_button()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not read file:\n{str(e)}")

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
            QMessageBox.warning(self, "Error", "Start Ollama server first!")
            return

        temp_file = "Modelfile_temp"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)
            self.log(f"🛠️ Creating model: {name}...")
            result = subprocess.run(["ollama", "create", name, "-f", temp_file], capture_output=True, text=True)
            os.remove(temp_file)
            if result.returncode == 0:
                self.log(f"✅ Custom model '{name}' created successfully!")
                self.load_models()
            else:
                self.log(f"❌ Create failed:\n{result.stderr}")
                QMessageBox.critical(self, "Error", result.stderr)
        except Exception as e:
            self.log(f"❌ {e}")

    def pull_model(self):
        model = self.pull_input.text().strip()
        if not model:
            QMessageBox.warning(self, "Error", "Enter a model name!")
            return
        if not self.server_ready:
            QMessageBox.warning(self, "Error", "Start server first!")
            return

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.log(f"⬇️ Pulling {model}...")

        def pull():
            try:
                proc = subprocess.Popen(["ollama", "pull", model], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.log(line.strip())
                proc.wait()
                self.log(f"✅ {model} pulled successfully!")
                self.load_models()
            except Exception as e:
                self.log(f"❌ Error: {e}")
            finally:
                self.progress.setVisible(False)

        threading.Thread(target=pull, daemon=True).start()

    def remove_model(self):
        if not hasattr(self, 'current_selected_model'):
            return
        model = self.current_selected_model
        reply = QMessageBox.question(self, "Confirm", f"Permanently delete '{model}'?")
        if reply == QMessageBox.Yes:
            try:
                subprocess.run(["ollama", "rm", model], check=True)
                self.log(f"🗑️ Removed {model}")
                self.load_models()
                self.selected_label.setText("No model selected")
                self.rm_btn.setEnabled(False)
            except Exception as e:
                self.log(f"❌ {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OllamaManager()
    win.show()
    sys.exit(app.exec_())
