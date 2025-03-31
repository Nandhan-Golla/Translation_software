from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class OfflineTranslator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def translate(self, text: str, target_language: str) -> str:
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        outputs = self.model.generate(inputs, num_beams=4, max_length=50, early_stopping=True)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text