import psycopg2
from psycopg2.extras import RealDictCursor
from config import DB_CONFIG


class PostgresDB:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.conn.autocommit = True

    # =========================================================
    # Conversations
    # =========================================================

    def create_conversation(self, title=None, model=None, folder=None):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (title, model, folder)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (title, model, folder)
            )
            return cur.fetchone()[0]

    def list_conversations(self, search=None, folder=None):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT id, title, model, created_at,
                       pinned, folder, summary
                FROM conversations
                WHERE 1=1
            """
            params = []

            if search:
                query += " AND title ILIKE %s"
                params.append(f"%{search}%")

            if folder:
                query += " AND folder = %s"
                params.append(folder)

            query += " ORDER BY pinned DESC, created_at DESC"

            cur.execute(query, params)
            return cur.fetchall()

    def get_conversation(self, conversation_id):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM conversations
                WHERE id = %s
                """,
                (conversation_id,)
            )
            return cur.fetchone()

    def rename_conversation(self, conversation_id, new_title):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET title = %s WHERE id = %s",
                (new_title, conversation_id)
            )

    def delete_conversation(self, conversation_id):
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE id = %s",
                (conversation_id,)
            )

    def toggle_pin(self, conversation_id):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE conversations
                SET pinned = NOT COALESCE(pinned, FALSE)
                WHERE id = %s
                """,
                (conversation_id,)
            )

    def set_folder(self, conversation_id, folder):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET folder = %s WHERE id = %s",
                (folder, conversation_id)
            )

    def set_conversation_summary(self, conversation_id, summary):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET summary = %s WHERE id = %s",
                (summary, conversation_id)
            )

    def clear_conversation(self, conversation_id):
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM messages WHERE conversation_id = %s",
                (conversation_id,)
            )

    # =========================================================
    # Messages
    # =========================================================

    def add_message(self, conversation_id, role, content):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (conversation_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (conversation_id, role, content)
            )

    def get_messages(self, conversation_id):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, role, content, created_at
                FROM messages
                WHERE conversation_id = %s
                ORDER BY id
                """,
                (conversation_id,)
            )
            return cur.fetchall()

    # =========================================================
    # Streaming Assistant (SAFE)
    # =========================================================

    def start_assistant_message(self, conversation_id):
        """
        Create empty assistant message and return message_id
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (conversation_id, role, content)
                VALUES (%s, 'assistant', '')
                RETURNING id
                """,
                (conversation_id,)
            )
            return cur.fetchone()[0]

    def append_message_chunk(self, message_id, chunk):
        """
        Append streaming token safely (NULL-safe)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE messages
                SET content = COALESCE(content, '') || %s
                WHERE id = %s
                """,
                (chunk, message_id)
            )

    def finalize_assistant_message(self, message_id, full_content):
        """
        Optional final overwrite (safety)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE messages
                SET content = %s
                WHERE id = %s
                """,
                (full_content, message_id)
            )

    # =========================================================
    # Export / Utility
    # =========================================================

    def export_conversation(self, conversation_id):
        """
        Returns conversation + messages (for markdown / pdf export)
        """
        convo = self.get_conversation(conversation_id)
        messages = self.get_messages(conversation_id)
        return {
            "conversation": convo,
            "messages": messages
        }

    def close(self):
        self.conn.close()
