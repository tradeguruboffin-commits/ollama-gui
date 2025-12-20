import subprocess

def handle(command: str):
    if command.startswith("run "):
        cmd = command[4:]
        try:
            out = subprocess.getoutput(cmd)
            return out
        except Exception as e:
            return str(e)
