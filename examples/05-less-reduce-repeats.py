import nltk
import numpy as np
import networkx as nx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

nltk.download("punkt_tab")
nltk.download("stopwords")

# Модель RuT5 для финального улучшения текста
rephrase_model = pipeline("text2text-generation", model="IlyaGusev/rut5_base_sum_gazeta")

def clean_text(text):
    """Удаляет лишние пробелы и символы"""
    return re.sub(r'\s+', ' ', text).strip()

def summarize_text(text, num_sentences=5):
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
    top_sentences = sorted([s for _, s in ranked_sentences[:num_sentences]])

    # 6. Соединяем их в связный текст
    summary = " ".join(top_sentences)

    # 7. Финально улучшаем текст через RuT5
    summary = rephrase_model(summary, max_length=300, num_return_sequences=1, num_beams=5, repetition_penalty=1.2)[0]["generated_text"]

    return clean_text(summary)

# Пример входного текста
text = """Ответ на вопрос «Почему небо голубое?» довольно прост. Когда свет от Солнца падает на Землю, он проходит по межпланетному пространству, представляющему собой вакуум, входит в атмосферу и взаимодействует с неоднородностями воздуха, в состав которого входят атомы разных элементов: кислорода, азота, углерода. На этих неоднородностях происходит рассеяние света, впервые рассмотренное Релеем. Этот процесс можно сравнить с полетом струи воды и попаданием ее на какую-либо решетку. Только природа взаимодействия здесь, конечно, другая.
Рассеяние света зависит от длины волны света. От Солнца идет широкий диапазон длин волн света, и он охватывает разные цвета: красный, оранжевый, желтый, зеленый, синий, фиолетовый. Эти цвета располагаются по порядку уменьшения длины волны: красный имеет наибольшую длину волны, желтый — меньшую, и фиолетовый имеет наименьшую длину. Оптический диапазон человеческого глаза может достигнуть лишь синего спектра волны света. Свет с короткой длиной волны рассеивается наиболее эффективно. А самая короткая длина волны для глаза — это синяя, поэтому синий свет заполняет все небо и мы видим его голубым. Другие цвета видимого спектра также рассеиваются, но гораздо меньше. Если бы человек мог видеть более короткие длины волн, чем та, которую имеет синий цвет, или, наоборот, не видел бы даже синей длины волны, то цвет неба для глаза был бы другой. Скажем, если бы мы видели только до зеленого цвета, а дальше наш глаз не видел, небо было бы зеленым. Смешение синего и фиолетового цветов находится на границе видимости глаза в коротковолновой части светового спектра и дает голубой оттенок. Каждый цвет имеет не одну длину волны, а диапазон длин волн, так что один цвет постепенно переходит в другой с изменением длины волны. Так, например, с уменьшением длины волны красный цвет постепенно переход в оранжевый, оранжевый в желтый, синий в фиолетовый.
Ночью небо темное, потому что Земля затмевает Солнце, и на ночной стороне земного шара мы видим только звезды. В этот момент в атмосфере ничего не рассеивается. По мере захода Солнца атмосфера Земли на ночной стороне получает все меньше рассеянного света. Скажем, в полночь мы находимся строго на линии центр Земли — Солнце, и Земля для нас полностью затмевает Солнце. Ни слева (с востока), ни справа (с запада) свет к нам в ночную атмосферу не попадает."""

# Если бы человек мог видеть более короткие длины волн, чем та, которая имеет синий цвет, то цвет неба для глаза был бы другой. Каждый цвет имеет не одну длину волны, а диапазон длин волн света, который охватывает разные цвета.

# Получаем итоговый пересказ
print(summarize_text(text))
