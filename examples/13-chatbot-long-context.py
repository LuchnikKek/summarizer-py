"""
Суммаризатор контекста.

Нормализует инпут в предложения.
Взвешивает предложения и закидывает в контекст num_sentences самых увесистых.
Никак не переписывает сами предложения.
"""

import heapq
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from storages import InMemoryStorage, IStorage


def _to_sentence_case(text: str) -> str:
    """Форматирует строку как предложение

    "пример фразы" -> "Пример фразы."
    "восклицание и вопрос остаётся!" -> "Восклицание и вопрос остаётся!"
    """
    text = text.strip()

    if not text:
        return text

    text = text[0].upper() + text[1:]

    if text[-1] not in ".?!":
        text += "."

    return text


class TextSummarizer:
    """Служебный класс, инкапсулирует логику суммаризации

    1. Очистка и токенизация
    Текст разбивается на слова, удаляются стоп-слова.

    2. Подсчёт частоты слов
    Каждому слову присваивается вес — чем чаще слово встречается, тем выше его вес.

    3.	Оценка значимости предложений
    Для каждого предложения суммируются веса всех содержащихся в нём слов.
    Чем больше значимых слов в предложении, тем выше его “оценка”.

    4.	Выбор ключевых предложений
    Из всех предложений выбираются num_sentences самых весомых (метод heapq.nlargest()).
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("russian"))

    def summarize(self, text: str, num_sentences: int = 3):
        # Токенизация текста на предложения
        sentences = sent_tokenize(text, language="russian")

        # Очистка текста и разбиение на слова
        word_frequencies = {}
        for word in word_tokenize(re.sub(r"[^а-яА-Я]", " ", text.lower()), language="russian"):
            if word not in self.stop_words:
                if word in word_frequencies:
                    word_frequencies[word] += 1
                else:
                    word_frequencies[word] = 1

        # Нормализация частот
        max_frequency = max(word_frequencies.values(), default=1)
        for word in word_frequencies:
            word_frequencies[word] /= max_frequency

        # Оценка предложений по их значимости
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower(), language="russian"):
                if word in word_frequencies:
                    if sentence in sentence_scores:
                        sentence_scores[sentence] += word_frequencies[word]
                    else:
                        sentence_scores[sentence] = word_frequencies[word]

        # Выбор ключевых предложений
        summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        summary = " ".join(summary_sentences)

        return summary


class TextSummarizerWithContext:
    """Суммаризатор с сохранением контекста

    storage - Хранилище контекстов. Из него контекст достается по айдишнику
    summarizer - Класс-суммаризатор
    """

    def __init__(self, summarizer: TextSummarizer, storage: IStorage):
        self.summarizer = summarizer
        self.storage = storage

    def normalize_text(self, text) -> str:
        return _to_sentence_case(text)

    def summarize_with_context(self, user_id, text, num_sentences=5):
        # Нормализация. Если предложение будет с маленькой буквы - склеивание будет неверным
        normalized_text = self.normalize_text(text)

        # Получаем контекст пользователя
        user_context = self.storage.load_context(user_id) or ""

        # Объединяем новый текст с контекстом
        combined_text = user_context + " " + normalized_text

        # Генерируем суммаризацию
        summary = self.summarizer.summarize(combined_text, num_sentences)

        # Сохраняем обновленный контекст пользователя
        self.storage.save_context(user_id, combined_text, ttl=3600)

        return summary


if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("stopwords")

    summarizer = TextSummarizerWithContext(
        summarizer=TextSummarizer(),
        storage=InMemoryStorage(),
    )

    user_id = "abc123"
    inputs = [
        "привет, меня зовут Алекс. Я работаю в Яндексе, люблю Python и играю в футбол",
        "мою маму зовут Люба. В 2023 году я начал стартап в области машинного обучения",
        "я вырос в Москве и окончил МГУ, факультет вычислительной математики и кибернетики",
        "в свободное время мне нравится путешествовать, особенно в горы",
        "я часто участвую в хакатонах и люблю разрабатывать новые идеи",
        "у меня есть кот по имени Барсик, который всегда сидит рядом, когда я программирую",
        "моя мечта — создать искусственный интеллект, который поможет людям в повседневной жизни",
        "я активно слежу за новыми технологиями и стараюсь применять их в своих проектах",
        "кроме футбола, я люблю настольные игры и шахматы",
        "в детстве я увлекался робототехникой и даже собирал небольшие модели роботов",
        "мне нравится читать книги по саморазвитию и бизнесу, особенно про стартапы и инвестиции.",
    ]

    for text in inputs:
        result = summarizer.summarize_with_context(user_id=user_id, text=text, num_sentences=5)

    print(result)
    # Моя мечта — создать искусственный интеллект, который поможет людям в повседневной жизни.
    # Мне нравится читать книги по саморазвитию и бизнесу, особенно про стартапы и инвестиции.
    # Я часто участвую в хакатонах и люблю разрабатывать новые идеи.
    # Я вырос в Москве и окончил МГУ, факультет вычислительной математики и кибернетики.
    # В свободное время мне нравится путешествовать, особенно в горы.
