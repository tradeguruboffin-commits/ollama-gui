#!/usr/bin/env python3
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from config import DB_CONFIG


class PostgresDB:
    def __init__(self):
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.conn.autocommit = True
            self._create_tables()
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise

    def _create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    model TEXT,
                    folder TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    pinned BOOLEAN DEFAULT FALSE,
                    summary TEXT
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)

            # NEW: Multi-Crew Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS crews (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    config TEXT NOT NULL,
                    is_default BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    # Legacy single crew (backward compatibility)
    def save_default_crew(self, config):
        config_json = json.dumps(config)
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO settings (key, value)
                VALUES ('default_crew', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
            """, (config_json,))

    def get_default_crew(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT value FROM settings WHERE key = 'default_crew';")
            row = cur.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except:
                    return []
        return []

    # ==================== MULTI-CREW FUNCTIONS ====================

    def create_crew(self, name, config, is_default=False):
        config_json = json.dumps(config)
        with self.conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO crews (name, config, is_default)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (name, config_json, is_default))
                return cur.fetchone()[0]
            except psycopg2.IntegrityError:
                raise ValueError(f"Crew '{name}' already exists!")

    def list_crews(self):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, name, config, is_default
                FROM crews
                ORDER BY is_default DESC, name ASC
            """)
            return cur.fetchall()

    def get_crew(self, crew_id):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM crews WHERE id = %s", (crew_id,))
            return cur.fetchone()

    def update_crew(self, crew_id, name, config):
        config_json = json.dumps(config)
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE crews SET name = %s, config = %s WHERE id = %s
            """, (name, config_json, crew_id))

    def delete_crew(self, crew_id):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM crews WHERE id = %s", (crew_id,))

    def set_default_crew(self, crew_id):
        with self.conn.cursor() as cur:
            cur.execute("UPDATE crews SET is_default = FALSE;")
            cur.execute("UPDATE crews SET is_default = TRUE WHERE id = %s;", (crew_id,))

    def get_default_crew_config(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT config FROM crews WHERE is_default = TRUE;")
            row = cur.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except:
                    pass
        # Fallback to legacy settings table
        return self.get_default_crew()

    # ==================== REST OF YOUR ORIGINAL FUNCTIONS ====================

    def create_conversation(self, title=None, model=None, folder=None):
        title = (title or "New Chat").strip()
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations (title, model, folder)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (title, model, folder))
            return cur.fetchone()[0]

    def list_conversations(self, search=None, folder=None):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT id, title, model, created_at, pinned, folder, summary FROM conversations WHERE 1=1"
            params = []
            if search:
                query += " AND title ILIKE %s"
                params.append(f"%{search.strip()}%")
            if folder:
                query += " AND folder = %s"
                params.append(folder)
            query += " ORDER BY pinned DESC, created_at DESC"
            cur.execute(query, params)
            return cur.fetchall()

    def get_conversation(self, conversation_id):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM conversations WHERE id = %s", (conversation_id,))
            return cur.fetchone()

    def rename_conversation(self, conversation_id, new_title):
        with self.conn.cursor() as cur:
            cur.execute("UPDATE conversations SET title = %s WHERE id = %s",
                        (new_title.strip(), conversation_id))

    def delete_conversation(self, conversation_id):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))

    def toggle_pin(self, conversation_id):
        with self.conn.cursor() as cur:
            cur.execute("UPDATE conversations SET pinned = NOT COALESCE(pinned, FALSE) WHERE id = %s", (conversation_id,))

    def add_message(self, conversation_id, role, content):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (%s, %s, %s)
            """, (conversation_id, role, str(content)))

    def get_messages(self, conversation_id):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, role, content, created_at
                FROM messages WHERE conversation_id = %s ORDER BY id ASC
            """, (conversation_id,))
            return cur.fetchall()

    def export_conversation(self, conversation_id):
        return {"conversation": self.get_conversation(conversation_id), "messages": self.get_messages(conversation_id)}

    def close(self):
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    db = PostgresDB()
    print("✅ Database ready!")
    print("Crews:", len(db.list_crews()))
    print("Default crew agents:", len(db.get_default_crew_config()))
    db.close()
