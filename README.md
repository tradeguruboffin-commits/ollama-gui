Ollama GUI UnofficialGitHub License
PyPI - Version
PyPI - Python VersionA feature-rich, unofficial Ollama GUI built with PyQt5, designed for advanced local AI interactions. This project extends the original Ollama capabilities with multi-crew agents, RAG support, and more. No additional dependencies beyond standard Python libraries and required packages.This is an Unofficial ForkThis is an enhanced, unofficial version of Ollama GUI, focusing on advanced features like crew workflows, RAG knowledge integration, and database-backed chat history. Ideal for developers and power users.
Install via Pi-Apps (if on Raspberry Pi) or follow the instructions below.
![badge](https://github.com/Botspot/pi-apps/blob/master/icons/badge.png?raw=true)ollama-gui-screenshot  <!-- আপনার স্ক্রিনশট লিঙ্ক অ্যাড করুন --> FeaturesCore Features Multi-file project with modular structure.
 Dependencies: PyQt5, requests, langchain (for RAG), PostgreSQL (for chat history).
 Auto-detect Ollama models and server status.
 Custom Ollama host support.
 Multiple conversations with search, pin, rename, and delete.
 Sidebar for chats and crews; right-click menus for management.
 Stop or reload generation at any time.

Advanced Features (v1.0+) Model Management: Integrate with separate Ollama Manager for pull/push/remove.
 UI Enhancements: Dark/light theme toggle, modern styling.
 Editable Conversation History with PostgreSQL backend.
 Crew Mode: Multi-agent workflows (e.g., Planner, Coder) for complex tasks.
 RAG Support: Add knowledge from PDFs, TXT, MD, DOCX, HTML; clear database.
 Image Attachments: Attach and send images in chats.
 Export Chats: Save as MD or TXT.
 Stats: Display generation stats (chunks streamed, speed).

 Before StartSet up Ollama service first. Refer to:  Ollama  
Ollama GitHub

Also, ensure PostgreSQL is installed and configured for chat history. Install required Python packages:  

pip install PyQt5 requests langchain langchain-community langchain-chroma psycopg2-binary

 RunChoose any method:Source Code

python ollama_gui.py

Using Pip (if published)

pip install ollama-gui-unofficial
ollama-gui-unofficial

Binary File (Coming Soon)Platform
Download Link
Windows
Download
Mac (Apple Silicon)
Download
Linux
Download

 Q&AWhy is the app slow when adding RAG documents?This can happen with large folders. Optimize by processing in batches, using smaller chunk sizes, or enabling GPU support in Ollama. See the optimization guide in the docs.ImportError: No module named 'PyQt5' or other dependencies?Install the required packages:  

pip install PyQt5 requests psycopg2-binary langchain langchain-community langchain-chroma langchain-ollama

For Ubuntu/Debian:  

sudo apt-get install python3-pyqt5 libpq-dev

For macOS:  

brew install pyqt postgresql

For Windows: Ensure PostgreSQL is installed and paths are set.Database connection issues?Configure PostgreSQL credentials in database/postgres.py. Create the database schema as per the code.Ollama server not responding?Ensure Ollama is running on localhost:11434. Start it with ollama serve.Refer to: https://stackoverflow.com/questions/related-to-dependencies for more troubleshooting.LicenseThis project is licensed under the MIT License (LICENSE).

