#! /usr/bin/env python
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorSearchTester:
    def __init__(self, 
                 index_path: str = "index",
                 model_name: str = "intfloat/Multilingual-E5-large"):
        self.index_path = Path(index_path)
        self.model_name = model_name

        print(f"Загрузка модели эмбеддингов: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.load_index()
    
    def load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Индекс не найден: {self.index_path}")
        
        # Загружаем метаданные (чанки)
        with open(self.index_path / "metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Загружаем FAISS индекс
        self.index = faiss.read_index(str(self.index_path / "faiss.index"))
        self.chunks = self.metadata.get("chunks", [])
        
        print(f"Индекс загружен: {len(self.chunks)} чанков")
    
    def search(self, query: str, k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        # Создаём эмбеддинг запроса
        query_vector = self.model.encode([query], convert_to_numpy=True)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Ищем ближайшие чанки
        scores, indices = self.index.search(query_vector.astype('float32'), k)
        
        # Форматируем результаты
        filtered_results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                relevance = float(score)
                
                if relevance >= threshold:
                    filtered_results.append({
                        'source_id': chunk.get('source_id', 'Unknown'),
                        'chunk_id': chunk.get('chunk_id', 'N/A'),
                        'text': chunk.get('text', ''),
                        'relevance': relevance
                    })
        
        return filtered_results
    
    def display_results(self, query: str, results: List[Dict[str, Any]], verbose: bool = True):
        print(f"\n{'=' * 60}")
        print(f"ЗАПРОС: {query}")
        print(f"{'=' * 60}")
        
        if not results:
            print("Релевантные документы не найдены")
            return
        
        print(f"Найдено релевантных чанков: {len(results)}\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Документ: {result['source_id']}.txt")
            print(f"   Релевантность: {result['relevance']:.4f}")
            print(f"   Чанк ID: {result['chunk_id']}")
            
            if verbose:
                text_preview = result['text'][:300]
                if len(result['text']) > 300:
                    text_preview += "..."
                print(f"   Текст:\n      {text_preview}\n")
            
            print("-" * 60)
    
    def test_golden_queries(self):
        golden_queries = [
            {
                "query": "Кто такой Рубикон нокс?",
                "description": "Вопрос о персонаже"
            },
            {
                "query": "Как появился на свет Леонтус Целестин?",
                "description": "Вопрос о происхождении"
            }
        ]
        
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ НА ЗОЛОТОМ НАБОРЕ ВОПРОСОВ")
        print("=" * 60)
        
        for test_case in golden_queries:
            query = test_case["query"]
            description = test_case["description"]
            
            print(f"\n Тест: {description}")
            print(f"   Запрос: {query}")
            
            results = self.search(query, k=3, threshold=0.5)
            
            if results:
                top_result = results[0]
                print(f"   Результат: НАЙДЕНО")
                print(f"   Найден: {top_result['source_id']}.txt (релевантность: {top_result['relevance']:.4f})")
            else:
                print(f"   Результат: НЕ НАЙДЕНО")
    
    def interactive_search(self):
        print("\n" + "=" * 60)
        print("ИНТЕРАКТИВНЫЙ ПОИСК")
        print("=" * 60)
        print("Введите запросы для поиска (или 'выход' для завершения)")
        
        while True:
            try:
                user_input = input("\n🔍 Запрос: ").strip()
                
                if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("Завершение работы...")
                    break
                
                if not user_input:
                    continue
                
                results = self.search(user_input, k=5, threshold=0.3)
                self.display_results(user_input, results)
                
            except KeyboardInterrupt:
                print("\n\nПрервано пользователем")
                break
            except Exception as e:
                print(f" Ошибка: {e}")


def main():
    parser = argparse.ArgumentParser(description='Тестирование векторного поиска')
    parser.add_argument('--index', default='index', help='Путь к индексу')
    parser.add_argument('--model', default='intfloat/Multilingual-E5-large',
                        help='Модель эмбеддингов')
    parser.add_argument('--query', help='Одиночный запрос для поиска')
    parser.add_argument('--k', type=int, default=5, help='Количество результатов')
    parser.add_argument('--test', action='store_true', 
                        help='Запустить тестирование на золотом наборе')
    parser.add_argument('--interactive', action='store_true', 
                        help='Интерактивный режим')
    
    args = parser.parse_args()

    tester = VectorSearchTester(args.index, args.model)
    
    if args.test:
        tester.test_golden_queries()
    elif args.query:
        results = tester.search(args.query, k=args.k)
        tester.display_results(args.query, results)
    elif args.interactive:
        tester.interactive_search()
    else:
        tester.interactive_search()


if __name__ == "__main__":
    main()