from db.postgres import PostgresDB

db = PostgresDB()

cid = db.create_conversation(
    title="First Test Chat",
    model="llama3"
)

db.add_message(cid, "user", "Hello Ollama!")
db.add_message(cid, "assistant", "Hello! How can I help you?")

messages = db.get_messages(cid)

for m in messages:
    print(f"[{m['role']}] {m['content']}")

db.close()
