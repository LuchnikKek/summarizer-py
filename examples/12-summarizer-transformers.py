from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TextSummarizer:
    def __init__(self, model_name="cointegrated/rut5-base-multitask"):
        """
        Используем модель T5, адаптированную под мультиязычные (включая русские) задачи:
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text: str, max_length: int = 100) -> str:
        """
        Для T5-моделей обычно используется префикс вроде 'summarize: ' перед исходным текстом.
        """
        input_text = f"summarize: {text}"
        inputs = self.tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt")
        output_ids = self.model.generate(
            **inputs, max_length=max_length, num_beams=4, no_repeat_ngram_size=3, early_stopping=True
        )
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return summary.strip()


if __name__ == "__main__":
    summarizer = TextSummarizer()
    print(summarizer.summarize("Привет, меня зовут Алекс. Я работаю в Яндексе, люблю Python и играю в футбол."))
    print(summarizer.summarize("Мою маму зовут Люба. В 2023 году я начал стартап в области машинного обучения."))
