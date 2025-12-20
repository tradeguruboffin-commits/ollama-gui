import subprocess
import os

def handle(command: str):
    if command.startswith("open "):
        path = command[5:].strip()
        if os.path.exists(path):
            subprocess.Popen(["xdg-open", path])
            return f"Opened: {path}"
        else:
            return f"File not found: {path}"
