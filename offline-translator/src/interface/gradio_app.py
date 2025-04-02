def create_gradio_interface():
    translator = Translator()
    
    def translate_text(text):
        return translator.translate(text)
    
    def train_model(src_text, tgt_text):
        src_sentences = src_text.split('\n')
        tgt_sentences = tgt_text.split('\n')
        if len(src_sentences) != len(tgt_sentences):
            return "Error: Number of source and target sentences must match"
        
        translator.train(src_sentences, tgt_sentences)
        torch.save(translator.model.state_dict(), 'translation_model.pth')
        return "Model trained and saved successfully"
    
    with gr.Blocks(title="Offline Translation System") as interface:
        gr.Markdown("# Offline Translation System")
        
        with gr.Tab("Translate"):
            text_input = gr.Textbox(label="Input Text")
            translate_button = gr.Button("Translate")
            output = gr.Textbox(label="Translated Text")
            translate_button.click(
                fn=translate_text,
                inputs=text_input,
                outputs=output
            )
        
        with gr.Tab("Train"):
            with gr.Row():
                src_input = gr.Textbox(label="Source Sentences (one per line)", lines=10)
                tgt_input = gr.Textbox(label="Target Sentences (one per line)", lines=10)
            train_button = gr.Button("Train Model")
            train_output = gr.Textbox(label="Training Status")
            train_button.click(
                fn=train_model,
                inputs=[src_input, tgt_input],
                outputs=train_output
            )
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()
