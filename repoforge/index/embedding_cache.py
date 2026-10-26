"""Small per-text embedding cache; successful batches survive interrupted builds."""
import hashlib
import json
from pathlib import Path
import sqlite3


def cache_key(settings, text):
    return hashlib.sha256(json.dumps([settings, text], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'embeddings.sqlite'

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL, checksum TEXT NOT NULL)')
        return connection

    def read(self, keys):
        connection = self._connect()
        try:
            result = {}
            for key in keys:
                row = connection.execute('SELECT vector, checksum FROM embeddings WHERE key = ?', (key,)).fetchone()
                if row:
                    if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
                        raise ValueError('Embedding cache checksum changed')
                    result[key] = json.loads(row[0])
            return result
        finally:
            connection.close()

    def write(self, rows):
        connection = self._connect()
        try:
            with connection:
                for key, vector in rows.items():
                    data = json.dumps(vector, separators=(',', ':'), allow_nan=False)
                    connection.execute('INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)',
                                       (key, data, hashlib.sha256(data.encode()).hexdigest()))
        finally:
            connection.close()
