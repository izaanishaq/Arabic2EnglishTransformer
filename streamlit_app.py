import streamlit as st
import torch
import re
import torch.nn as nn
from spacy.lang.ar import Arabic
from spacy.tokenizer import Tokenizer
import numpy.core.multiarray as multiarray

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize Spacy Arabic tokenizer
ar_nlp = Arabic()
ar_tokenizer_obj = Tokenizer(ar_nlp.vocab)

def myTokenizerAR(text):
    # Clean and tokenize Arabic text.
    text = re.sub(r"[\.\'\`\"\r\n+]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return [token.text for token in ar_tokenizer_obj(text)]

# Function to rebuild the vocabulary object
def load_vocab_from_tokens(token_list):
    Vocab = type("Vocab", (), {})
    vocab = Vocab()
    vocab.stoi = {token: i for i, token in enumerate(token_list)}
    vocab.itos = {i: token for i, token in enumerate(token_list)}
    vocab.size = len(token_list)
    return vocab

# Define the translation transformer model
class TranslateTransformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        max_len,
    ):
        super(TranslateTransformer, self).__init__()
        self.srcEmbeddings = nn.Embedding(src_vocab_size, embedding_size)
        self.trgEmbeddings = nn.Embedding(trg_vocab_size, embedding_size)
        self.srcPositionalEmbeddings = nn.Embedding(max_len, embedding_size)
        self.trgPositionalEmbeddings = nn.Embedding(max_len, embedding_size)
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.src_pad_idx = src_pad_idx
        self.max_len = max_len
    
    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask.to(device)
    
    def forward(self, x, trg):
        src_seq_length = x.shape[0]
        batch_size = x.shape[1]
        trg_seq_length = trg.shape[0]
        
        src_positions = torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, batch_size).to(device)
        trg_positions = torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, batch_size).to(device)
        
        src_embedded = self.srcEmbeddings(x.long()) + self.srcPositionalEmbeddings(src_positions.long())
        trg_embedded = self.trgEmbeddings(trg.long()) + self.trgPositionalEmbeddings(trg_positions.long())
        
        src_embedded = self.dropout(src_embedded)
        trg_embedded = self.dropout(trg_embedded)
        
        src_padding_mask = self.make_src_mask(x)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(device)
        
        out = self.transformer(
            src_embedded,
            trg_embedded,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask
        )
        out = self.fc_out(out)
        return out

# Load the checkpoint
with torch.serialization.safe_globals([multiarray.scalar]):
    checkpoint = torch.load("model_checkpoint.pt", map_location=device, weights_only=False)
config = checkpoint["config"]

# Reassemble vocabularies using the saved token lists.
src_vocab = load_vocab_from_tokens(checkpoint["src_vocab_tokens"])
tgt_vocab = load_vocab_from_tokens(checkpoint["tgt_vocab_tokens"])
src_pad_idx = src_vocab.stoi["<pad>"]

# Re-create the model architecture and load state.
model = TranslateTransformer(
    config["embedding_size"],
    src_vocab.size,
    tgt_vocab.size,
    src_pad_idx,
    config["num_heads"],
    config["num_encoder_layers"],
    config["num_decoder_layers"],
    config["max_len"]
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def translate(model, sentence, src_tokenizer, src_vocab, tgt_vocab, max_len=200, beam_size=3):
    model.eval()
    # Prepare source sentence with start and end tokens.
    tokens = ["ببدأ"] + src_tokenizer(sentence) + ["نهها"]
    src_indices = [src_vocab.stoi.get(tok, src_vocab.stoi["<unk>"]) for tok in tokens]
    src_tensor = torch.tensor(src_indices).unsqueeze(1).to(device)  # [src_len, 1]

    # Initialize beam with the <sos> token.
    initial_idx = tgt_vocab.stoi["<sos>"]
    beams = [([initial_idx], 0)]
    
    for _ in range(max_len):
        new_beams = []
        for trg_sequence, score in beams:
            # If sequence completed, keep it.
            if trg_sequence[-1] == tgt_vocab.stoi["<eos>"]:
                new_beams.append((trg_sequence, score))
                continue
            # Prepare target tensor and generate model outputs.
            trg_tensor = torch.tensor(trg_sequence).unsqueeze(1).to(device)
            output = model(src_tensor, trg_tensor)
            # Consider the last token's output.
            logits = output[-1].squeeze(0)  # [vocab_size]
            log_probs = torch.log_softmax(logits, dim=0)
            # Select top 'beam_size' candidates
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
            for log_prob, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                new_seq = trg_sequence + [idx]
                new_beams.append((new_seq, score + log_prob))
        # Retain the best beam_size sequences.
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        # Stop if all beams end with <eos>.
        if all(seq[-1] == tgt_vocab.stoi["<eos>"] for seq, _ in beams):
            break
    
    best_seq = beams[0][0]
    # Remove <sos> and trim <eos> if present.
    translated_tokens = [tgt_vocab.itos[idx] for idx in best_seq if idx != tgt_vocab.stoi["<sos>"]]
    if translated_tokens and translated_tokens[-1] == "<eos>":
        translated_tokens = translated_tokens[:-1]
    return " ".join(translated_tokens)

# Streamlit UI
st.title("Arabic to English Translator")
st.write("Enter an Arabic sentence to translate:")

arabic_input = st.text_input("Arabic Input", "مرحبًا")

if st.button("Translate"):
    translation = translate(model, arabic_input, myTokenizerAR, src_vocab, tgt_vocab)
    st.write("**Translation (English):**", translation)