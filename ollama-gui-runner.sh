#!/bin/bash
python3 /opt/ollama-gui/ollama_gui.py

#stop running models to free up RAM immediately
IFS=$'\n'
for i in $(ollama ps | tail -n +2 | awk '{print $1}') ;do
  ollama stop "$i"
done
