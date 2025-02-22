import json
import sqlite3
from collections import defaultdict
from typing import Any, Dict

import dateparser
import spacy

nlp = spacy.load("ru_core_news_md")


class ShortMemoryStorage:
    """Кратковременная память (в оперативной памяти)"""

    def __init__(self):
        self.memory = defaultdict(dict)

    def save_context(self, user_id: str, context: Dict[str, Any]) -> None:
        """Сохраняет контекст пользователя в оперативной памяти"""
        self.memory[user_id] = context

    def load_context(self, user_id: str) -> Dict[str, Any]:
        """Загружает контекст пользователя"""
        return self.memory.get(user_id, {})

    def clear_context(self, user_id: str) -> None:
        """Очищает контекст пользователя"""
        self.memory.pop(user_id, None)


class LongMemoryStorage:
    """Долговременная память (хранение предпочтений пользователя)"""

    def __init__(self):
        self.conn = sqlite3.connect("long_term_memory.db")
        self.cursor = self.conn.cursor()

        # Создание таблицы, если она еще не создана
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            preferences TEXT
        )
        """)
        self.conn.commit()

    def save_context(self, user_id: str, new_prefs: Dict[str, Any]) -> None:
        """Сохраняет или обновляет долгосрочные предпочтения пользователя"""
        old_prefs = self.load_context(user_id)
        updated_prefs = {**old_prefs, **new_prefs}

        self.cursor.execute(
            """
            INSERT INTO user_preferences (user_id, preferences)
            VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET preferences = ?
            """,
            (user_id, json.dumps(updated_prefs), json.dumps(updated_prefs)),
        )
        self.conn.commit()

    def load_context(self, user_id: str) -> Dict[str, Any]:
        """Загружает предпочтения пользователя из базы данных"""
        self.cursor.execute("SELECT preferences FROM user_preferences WHERE user_id = ?", (user_id,))
        row = self.cursor.fetchone()
        return json.loads(row[0]) if row else {}


short_memory = ShortMemoryStorage()
long_memory = LongMemoryStorage()


def extract_entities(message: str) -> Dict[str, str]:
    """Извлекает сущности (город, дата) из сообщения пользователя"""
    doc = nlp(message)
    entities = {}

    # Извлекаем географические названия (города, страны)
    for ent in doc.ents:
        if ent.label_ in ["LOC", "GPE"]:  # Географическое название (город, страна)
            entities["location"] = ent.text

    # Парсим дату через dateparser
    parsed_date = dateparser.parse(message, languages=["ru"], settings={"SKIP_TOKENS": ["на", "в", "к", "за"]})
    if parsed_date:
        entities["date"] = parsed_date.strftime("%d.%m.%Y")  # Приводим к формату DD.MM.YYYY

    return entities


def chatbot_response(user_id: str, message: str) -> str:
    """Обрабатывает сообщение пользователя, используя кратковременную и долговременную память"""
    # Загружаем текущий контекст диалога и долгосрочные предпочтения
    short_term_context = short_memory.load_context(user_id)
    long_term_context = long_memory.load_context(user_id)

    response = ""

    # Простая логика: если пользователь упоминает отель, запрашиваем данные
    if "отель" in message.lower():
        if "location" in short_term_context and "date" in short_term_context:
            location = short_term_context["location"]
            date = short_term_context["date"]
            response = f"Бронирую отель в {location} на {date}!"

            # Сохраняем бронирование в долговременную память
            long_memory.save_context(user_id, {"last_booking": {"location": location, "date": date}})
            short_memory.clear_context(user_id)  # Очищаем кратковременный контекст
        else:
            response = "В каком городе и на какие даты вам нужен отель?"
            short_memory.save_context(user_id, {"intent": "book_hotel"})

    # Если пользователь спрашивает рекомендацию
    elif "рекомендуй" in message.lower():
        last_booking = long_term_context.get("last_booking")
        if last_booking:
            response = f"Вы ранее бронировали отель в {last_booking['location']}. Хотите снова?"
        else:
            response = "У вас ещё нет истории бронирования. Какой город вас интересует?"

    # Если пользователь уточняет детали бронирования
    elif short_term_context.get("intent") == "book_hotel":
        entities = extract_entities(message)

        if "location" in entities:
            short_memory.save_context(user_id, {**short_term_context, "location": entities["location"]})
            response = "На какие даты вам нужен отель?"
        elif "date" in entities:
            short_memory.save_context(user_id, {**short_term_context, "date": entities["date"]})
            response = f"Понял, бронирую отель на {entities['date']}!"
        else:
            response = "Не совсем понял. Можете уточнить город или дату?"

    else:
        response = "Я не понял ваш запрос. Вы хотите забронировать отель?"

    return response


if __name__ == "__main__":
    user_id = "user_123"
    print(chatbot_response(user_id, "Я хочу забронировать отель"))
    # В каком городе и на какие даты вам нужен отель?
    print(chatbot_response(user_id, "В Москве"))
    # На какие даты вам нужен отель?
    print(chatbot_response(user_id, "На 10 марта"))
    # Понял, бронирую отель на 10.03.2025!
    print(chatbot_response(user_id, "Рекомендуй мне"))
    # Вы ранее бронировали отель в Москве. Хотите снова?
