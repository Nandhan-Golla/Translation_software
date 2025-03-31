from transformers import pipeline

class OfflineTranslator:
    def __init__(self, model_name='Helsinki-NLP/opus-mt-en-fr'):
        self.translator = pipeline('translation', model=model_name)

    def translate(self, text, target_language='fr'):
        translation = self.translator(text, target_lang=target_language)
        return translation[0]['translation_text'] if translation else None

if __name__ == "__main__":
    translator = OfflineTranslator()
    sample_text = "Hello, how are you?"
    translated_text = translator.translate(sample_text)
    print(f"Translated Text: {translated_text}")