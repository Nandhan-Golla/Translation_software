
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from typing import List, Tuple
import numpy as np
from ollama import Client
import json
import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key="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")

class ComplexAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q1 = nn.Linear(d_model, d_model)
        self.W_q2 = nn.Linear(d_model, d_model)
        self.W_k1 = nn.Linear(d_model, d_model)
        self.W_k2 = nn.Linear(d_model, d_model)
        self.W_v1 = nn.Linear(d_model, d_model)
        self.W_v2 = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q1 = self.W_q1(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        q2 = self.W_q2(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k1 = self.W_k1(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k2 = self.W_k2(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v1 = self.W_v1(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v2 = self.W_v2(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores1 = scores1.masked_fill(mask == 0, -1e9)
            scores2 = scores2.masked_fill(mask == 0, -1e9)
        
        attn1 = F.softmax(scores1, dim=-1)
        attn2 = F.softmax(scores2, dim=-1)
        
        context1 = torch.matmul(self.dropout(attn1), v1)
        context2 = torch.matmul(self.dropout(attn2), v2)
        
        context = torch.cat([context1, context2], dim=-1)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.output_layer(context)
        return self.norm(output)

class EnhancedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = ComplexAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff//2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        return self.norm2(x + self.dropout(ff_output))
        
class OllamaTranslationModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_pos_encoding(1000, d_model)
        
        self.encoder = nn.ModuleList([
            EnhancedTransformerBlock(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            EnhancedTransformerBlock(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def create_pos_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1)].to(src.device)
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1)].to(tgt.device)
        
        memory = src_emb
        for layer in self.encoder:
            memory = layer(memory, src_mask)
            
        output = tgt_emb
        for layer in self.decoder:
            output = layer(output, tgt_mask)
            
        return self.output_layer(output)

class OllamaTranslator:
    def __init__(self, ollama_model: str = "llama2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ollama_client = Client(host='http://localhost:11434')
        self.model = OllamaTranslationModel(vocab_size=50000).to(self.device)
        self.vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.inv_vocab = {0: '<pad>', 1: '<sos>', 2: '<eos>'}
        self.ollama_model_name = ollama_model
        
    def tokenize(self, text: str) -> List[str]:
        response = self.ollama_client.generate(
            model=self.ollama_model_name,
            prompt=f"Tokenize this text: {text}",
            system="Return space-separated tokens"
        )
        return response['response'].split()
    
    def build_vocab(self, sentences: List[str]):
        for sent in sentences:
            for token in self.tokenize(sent):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    self.inv_vocab[len(self.inv_vocab)] = token
    
    def text_to_tensor(self, text: str) -> torch.Tensor:
        tokens = ['<sos>'] + self.tokenize(text) + ['<eos>']
        return torch.tensor([self.vocab.get(t, 0) for t in tokens], 
                          dtype=torch.long).unsqueeze(0).to(self.device)
    
    def train(self, src_sentences: List[str], tgt_sentences: List[str], epochs: int = 10):
        self.build_vocab(src_sentences + tgt_sentences)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        for epoch in range(epochs):
            total_loss = 0
            for src, tgt in zip(src_sentences, tgt_sentences):
                src_tensor = self.text_to_tensor(src)
                tgt_tensor = self.text_to_tensor(tgt)
                
                optimizer.zero_grad()
                output = self.model(src_tensor, tgt_tensor[:, :-1])
                loss = criterion(output.view(-1, len(self.vocab)), 
                               tgt_tensor[:, 1:].view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(src_sentences)}")
    
    def translate(self, text: str, max_length: int = 50) -> str:
        self.model.eval()
        src_tensor = self.text_to_tensor(text)
        tgt_tensor = torch.tensor([[self.vocab['<sos>']]], 
                                dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                output = self.model(src_tensor, tgt_tensor)
                next_token = output[:, -1, :].argmax(-1)
                tgt_tensor = torch.cat([tgt_tensor, next_token.unsqueeze(-1)], dim=-1)
                
                if next_token.item() == self.vocab['<eos>']:
                    break
        
        translation = ' '.join(self.inv_vocab.get(idx.item(), '<unk>') 
                             for idx in tgt_tensor[0][1:-1])
        ollama_response = self.ollama_client.generate(
            model=self.ollama_model_name,
            prompt=f"Refine this translation: {translation}",
            system="Improve grammar and fluency"
        )
        return ollama_response['response']

def create_interface():
    translator = OllamaTranslator()
    
    def translate_text(text):
        return translator.translate(text)
    
    def train_model(src_text, tgt_text):
        src_sentences = src_text.split('\n')
        tgt_sentences = tgt_text.split('\n')
        if len(src_sentences) != len(tgt_sentences):
            return "Error: Number of source and target sentences must match"
        translator.train(src_sentences, tgt_sentences)
        torch.save(translator.model.state_dict(), 'ollama_translator.pth')
        return "Model trained and saved successfully"
    
    with gr.Blocks(title="Ollama Offline Translator") as interface:
        gr.Markdown("# Offline Translation with Local Ollama")
        with gr.Tab("Translate"):
            text_input = gr.Textbox(label="Input Text")
            translate_btn = gr.Button("Translate")
            output = gr.Textbox(label="Translated Text")
            translate_btn.click(translate_text, inputs=text_input, outputs=output)
        
        with gr.Tab("Train"):
            with gr.Row():
                src_input = gr.Textbox(label="Source Sentences", lines=10)
                tgt_input = gr.Textbox(label="Target Sentences", lines=10)
            train_btn = gr.Button("Train Model")
            train_output = gr.Textbox(label="Training Status")
            train_btn.click(train_model, inputs=[src_input, tgt_input], outputs=train_output)
    
    return interface

if __name__ == "__main__":

    try:
        ollama_client = Client(host='http://localhost:11434')
        ollama_client.list()
    except Exception as e:
        print("Error: Please ensure Ollama is running locally on port 11434")
        exit(1)
    
    interface = create_interface()
    interface.launch()
