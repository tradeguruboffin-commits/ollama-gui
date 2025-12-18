import subprocess, os

def run(cmd):
    return subprocess.getoutput(cmd)

def open_file(path):
    os.system(f"xdg-open '{path}'")
