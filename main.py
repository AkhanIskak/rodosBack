from flask import Flask, request, jsonify
import requests
import os
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModel
import torch
import json

app = Flask(__name__)

# Настройка API ключа и базового URL для Llama API
API_KEY = 'a413a4be80bb4d198772f1bce5c88bcd'
API_BASE_URL = "https://api.aimlapi.com"

# Инициализация ChromaDB
client = chromadb.Client()
collection_name = "bureaucracy_docs"
collection = client.get_or_create_collection(name=collection_name)

# Инициализация модели эмбеддинга
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def embed_text(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings.tolist()

# Эндпоинт для добавления данных
@app.route('/add_data', methods=['POST'])
def add_data():
    data = request.json
    documents = data.get('documents', [])
    ids = data.get('ids', [])
    metadatas = data.get('metadatas', [])

    if not documents or not ids:
        return jsonify({"status": "error", "message": "Поля 'documents' и 'ids' обязательны."}), 400

    # Преобразование документов в эмбеддинги
    embeddings = embed_text(documents)

    # Добавление документов в коллекцию ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return jsonify({"status": "success", "message": "Данные успешно добавлены в векторную базу данных."}), 200

# Эндпоинт для получения данных (генерация ответа с помощью Llama)
@app.route('/get_data', methods=['POST'])
def get_data():
    # Получение данных из запроса
    data = request.json
    user_prompt = data.get('user_prompt', '')
    system_prompt = data.get('system_prompt', 'Вы выступаете в роли помощника по государственным услугам.')

    if not user_prompt:
        return jsonify({"status": "error", "message": "Поле 'user_prompt' обязательно."}), 400

    try:
        # Преобразование запроса пользователя в эмбеддинг
        query_embedding = embed_text([user_prompt])

        # Поиск похожих документов в ChromaDB
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            include=['documents']
        )

        # Получение найденных документов
        relevant_docs = results['documents'][0] if results['documents'] else []

        # Подготовка prompt для Llama
        prompt = f"""{system_prompt}

Используя следующую информацию, ответьте на вопрос пользователя максимально подробно и точно.

Вопрос пользователя:
"{user_prompt}"

Информация:
"""
        for idx, doc in enumerate(relevant_docs, 1):
            prompt += f"{idx}. {doc}\n"
        # Вызов Llama API для генерации ответа
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "meta-llama/Meta-Llama-3-70B-Instruct-Lite",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": False,
        }

        response = requests.post(
            url=f"{API_BASE_URL}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
        )

        response.raise_for_status()
        completion = response.json()

        # Получение сгенерированного ответа
        generated_response = completion['choices'][0]['message']['content']

        return jsonify({"response": completion}), 200

    except Exception as e:
        # Обработка ошибок и возврат сообщения об ошибке
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Запуск приложения Flask
    app.run(host='0.0.0.0', port=3000, debug=True)