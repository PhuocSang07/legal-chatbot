import psycopg2
from dotenv import load_dotenv
import os 

load_dotenv()
pw = os.getenv("DB_PASSWORD")

class DatabaseInspector:
    def __init__(self, host, port, dbname, user, password):
        self.conn_string = (
            f"host={host} port={port} dbname={dbname} user={user} password={password}"
        )
        self.conn = None
        self.cursor = None

    def connect(self):
        self.conn = psycopg2.connect(self.conn_string)
        self.cursor = self.conn.cursor()

    def close(self):
        if self.cursor is not None:
            self.cursor.close()
        if self.conn is not None:
            self.conn.close()

    def table_exists(self, table_name):
        self.connect()
        try:
            self.cursor.execute(
                f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s);",
                (table_name,),
            )
            exists = self.cursor.fetchone()[0]
            return exists
        finally:
            self.close()

    def print_row_counts(self, table_names):
        for table_name in table_names:
            if self.table_exists(table_name):
                self.connect()
                try:
                    self.cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                    count = self.cursor.fetchone()[0]
                    print(f"Table '{table_name}' has {count} rows.")
                except Exception as e:
                    print(f"Error occurred while counting rows in '{table_name}': {e}")
                finally:
                    self.close()
            else:
                print(f"Table '{table_name}' not found.")


if __name__ == "__main__":
    cleaner = DatabaseInspector("localhost", "5432", "demo_bk", "postgres", pw)
    cleaner.clear_table_contents(["test_bk"])