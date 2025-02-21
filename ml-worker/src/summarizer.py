import abc
import re

import networkx as nx
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

nltk.download("punkt_tab")
nltk.download("stopwords")


class ISummarizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def summarize(self, text: str) -> str:
        """Возвращает суммаризацию текста"""
        raise NotImplementedError


class TFTextRankSummarizer(ISummarizer):
    """Суммаризатор с использование TF-IDF + TextRank + RuT5"""

    # Модель RuT5 для финального улучшения текста
    rephrase_model = pipeline("text2text-generation", model="IlyaGusev/rut5_base_sum_gazeta")
    num_sentences = 5

    def summarize(self, text: str) -> str:
        # 1. Разбиваем текст на предложения
        sentences = sent_tokenize(text)

        # 2. Вычисляем TF-IDF
        vectorizer = TfidfVectorizer(stop_words=stopwords.words("russian"))
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # 3. Строим граф предложений (по косинусному сходству)
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        np.fill_diagonal(similarity_matrix, 0)

        # 4. Применяем TextRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        # 5. Выбираем ТОП-5 предложений
        top_sentences = sorted([s for _, s in ranked_sentences[: self.num_sentences]])

        # 6. Соединяем их в связный текст
        summary = " ".join(top_sentences)

        # 7. Финально улучшаем текст через RuT5
        summary = self.rephrase_model(
            summary,
            max_length=300,
            num_return_sequences=1,
            num_beams=5,
            repetition_penalty=1.2,
        )[0]["generated_text"]

        return clean_text(summary)


def clean_text(text):
    """Удаляет лишние пробелы и символы"""
    return re.sub(r"\s+", " ", text).strip()
