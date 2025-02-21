import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

nltk.download("punkt_tab")
nltk.download("stopwords")

def summarize_text(text, num_sentences=3):
    # 1. Разбиваем на предложения
    sentences = sent_tokenize(text)
    
    # 2. Вычисляем TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stopwords.words("russian"))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # 3. Строим граф предложений (по косинусному сходству)
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    np.fill_diagonal(similarity_matrix, 0)  # Обнуляем связь предложения с самим собой
    
    # 4. Применяем TextRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    
    # 5. Выбираем топ-N предложений
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = " ".join([s for _, s in ranked_sentences[:num_sentences]])
    
    return summary

# Пример использования
text = """Данный алгоритм очень прост как для понимания, так и дальнейшей реализации. Здесь мы работаем только с исходным текстом, и по большому счету у нас отсутствует потребность в обучении какой-либо модели извлечения. В моем случае извлекаемые информационные блоки будут представлять собой определенные предложения текста.
Итак, на первом шаге разбиваем входной текст на предложения и каждое предложение разбиваем на токены (отдельные слова), проводим для них лемматизацию (приведение слова к «канонической» форме). Этот шаг необходим для того, чтобы алгоритм объединял одинаковые по смыслу слова, но отличающиеся по словоформам.
Затем задаем функцию схожести для каждой пары предложений. Она будет рассчитываться как отношение числа общих слов, встречающихся в обоих предложениях, к их суммарной длине. В результате мы получим коэффициенты схожести для каждой пары предложений.
Предварительно отсеяв предложения, которые не имеют общих слов с другими, построим граф, где вершинами являются сами предложения, ребра между которыми показывают наличие в них общих слов.."""

print(summarize_text(text, num_sentences=2))