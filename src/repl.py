#! /usr/bin/env python

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env из родительской папки (где лежит папка src)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

from config import INDEX_DIR, BASE_DIR
print(f"BASE_DIR = {BASE_DIR}")
print(f"INDEX_DIR = {INDEX_DIR}")
print(f"Индекс существует: {Path(INDEX_DIR).exists()}")
print(f"Загружен .env из: {env_path}")
print(f"YANDEX_API_KEY: {os.getenv('YANDEX_API_KEY')[:20] if os.getenv('YANDEX_API_KEY') else 'None'}...")

# Добавляем текущую папку в путь
sys.path.insert(0, str(Path(__file__).parent))

from rag_bot import RAGBot

bot = RAGBot()

def main():
    print("RAG REPL. Введите 'exit' для выхода.")
    while True:
        q = input("Q: ").strip()
        if q in ("exit", "quit"):
            break
        resp = bot.answer(q)
        print("---\nANSWER:\n", resp["answer"])
        print("SOURCES:", resp["source"])
        print("EXPLAIN:", resp["explain"])
        print("-----\n")

if __name__ == "__main__":
    main()