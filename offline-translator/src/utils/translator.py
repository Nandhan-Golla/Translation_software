class Translator:
    def __init__(self, src_lang: str = 'en', tgt_lang: str = 'es'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.src_tokenizer = get_tokenizer('spacy', language=src_lang)
        self.tgt_tokenizer = get_tokenizer('spacy', language=tgt_lang)
      
        self.model = TranslationModel(
            src_vocab_size=10000,
            tgt_vocab_size=10000
        ).to(self.device)
        
        self.src_vocab = {}
        self.tgt_vocab = {}
        self.inv_tgt_vocab = {} 
        
    def train(self, src_sentences: List[str], tgt_sentences: List[str], epochs: int = 10):
        self.build_vocabulary(src_sentences, tgt_sentences)
      
        self.model = TranslationModel(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 for padding
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for src, tgt in zip(src_sentences, tgt_sentences):
                src_tensor = self.text_to_tensor(src, self.src_vocab, self.src_tokenizer)
                tgt_tensor = self.text_to_tensor(tgt, self.tgt_vocab, self.tgt_tokenizer)
                
                optimizer.zero_grad()
                output = self.model(src_tensor, tgt_tensor[:, :-1])
                
                loss = criterion(output.view(-1, len(self.tgt_vocab)), 
                               tgt_tensor[:, 1:].view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(src_sentences)}")
            
    def build_vocabulary(self, src_sentences: List[str], tgt_sentences: List[str]):
        self.src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        self.tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        
        for sent in src_sentences:
            for token in self.src_tokenizer(sent):
                if token not in self.src_vocab:
                    self.src_vocab[token] = len(self.src_vocab)
                    
        for sent in tgt_sentences:
            for token in self.tgt_tokenizer(sent):
                if token not in self.tgt_vocab:
                    self.tgt_vocab[token] = len(self.tgt_vocab)
                    
        self.inv_tgt_vocab = {v: k for k, v in self.tgt_vocab.items()}
        
    def text_to_tensor(self, text: str, vocab: dict, tokenizer) -> torch.Tensor:
        tokens = ['<sos>'] + tokenizer(text) + ['<eos>']
        return torch.tensor([vocab.get(token, 0) for token in tokens], 
                          dtype=torch.long).unsqueeze(0).to(self.device)
    
    def translate(self, text: str) -> str:
        self.model.eval()
        with torch.no_grad():
            src_tensor = self.text_to_tensor(text, self.src_vocab, self.src_tokenizer)
            tgt_tensor = torch.tensor([[self.tgt_vocab['<sos>']]], 
                                    dtype=torch.long).to(self.device)
            
            for _ in range(50):
                output = self.model(src_tensor, tgt_tensor)
                next_token = output[:, -1, :].argmax(-1)
                tgt_tensor = torch.cat([tgt_tensor, next_token.unsqueeze(-1)], dim=-1)
                
                if next_token.item() == self.tgt_vocab['<eos>']:
                    break
            
            translation = ' '.join(self.inv_tgt_vocab.get(idx.item(), '<unk>') 
                                 for idx in tgt_tensor[0][1:-1])
            return translation
