"""
Пример шаблонизации с сохранением контекста
"""

import spacy
from storages import InMemoryStorage

nlp = spacy.load("ru_core_news_sm")


def extract_entities(text) -> dict[str, str]:
    # Функция для извлечения сущностей из текста
    doc = nlp(text)
    return {ent.label_: ent.text for ent in doc.ents}


def generate_response(user_id, message, memory) -> str:
    # Функция для генерации ответа
    context = memory.load_context(user_id)
    entities = extract_entities(message)
    context.update(entities)
    memory.save_context(user_id, context)

    return f"Рад тебя видеть, {context.get('PER', 'друг')} из {context.get('LOC', 'неизвестного места')}!"


if __name__ == "__main__":
    memory = InMemoryStorage()

    user_id = "123"
    message = "Привет, я Илья!"
    print(generate_response(user_id, message, memory))  # "Рад тебя видеть, Илья из неизвестного места!"

    message = "Я работаю программистом."
    print(generate_response(user_id, message, memory))  # "Рад тебя видеть, Илья из неизвестного места!"

    message = "Я из Москвы."
    print(generate_response(user_id, message, memory))  # "Рад тебя видеть, Илья из Москвы!"
