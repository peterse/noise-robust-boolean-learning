import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.nn import functional as F

class TokenEmbedding(nn.Module):
    """from https://pytorch.org/tutorials/beginner/translation_transformer.html
    
        Args:
            vocab_size: (int) number of tokens in alphabet
            emb_size: (int) model dimension
    """
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        """
        Input:
            tokens: (batch_size, m) tensor of bits or token indices (m=n or 2n)
        Returns:
            Tensor: (batch_size, n, emb_size), final dimension indexes the embedding vector
        """
        # Okay, so we have to cast our bits to float64 to embed...
        # FIXME: use lower precision?
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    """from https://pytorch.org/tutorials/beginner/translation_transformer.html
    
    Note: This has been heavily modified for BATCH FIRST mode.

    Args:
        emb_size: dimension of the embedding, i.e. d_model. MUST BE EVEN
        dropout: dropout rate
    """
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        # this just rearranges the equation from Vaswani et al. (2017)
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        # insert batch dimension up front for batch_first convention
        pos_embedding = pos_embedding.unsqueeze(0) # (1, maxlen, emb_size)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        """
        Input:
            token_embedding: (batch_size, n, emb_size)
        Returns:
            Tensor: (batch_size, n, emb_size), with positional encoding
        """
        # NOTE: dropout has a normalization subroutine so this object might 
        # have a weird norm. For instance, if the token embedding is all zeros you 
        # might get values larger than 1 (the maximum of sin, cos)
        sliced = self.pos_embedding[:, :token_embedding.size(-2)] # (1, sequence_len, emb_size)
        return self.dropout(token_embedding + sliced)
    

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Seq2SeqTransformer(nn.Module):
    """from https://pytorch.org/tutorials/beginner/translation_transformer.html"""
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size,
                 dim_feedforward=512, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True) # (batch, seq_len, d_model)
        
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        # Final layer for output decoder
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        """
        Let S be the source seq length, T the target seq length, N the batch size, E the embedding dimension.

        Args:
            src: input token embeddings. Shape: (N,S,E) (since Transformer.batch_first=True)
            trg: target token embeddings. Shape: (N,T,E) 
            src_mask: Encoder self-attention mask. Shape is (S,S) or (N⋅num_heads,S,S)
            tgt_mask: Decoder self-attention mask. Shape is (T,T) or (N⋅num_heads,T,T)
            src_padding_mask: This removes padding for ragged seqences, specified per example
            tgt_padding_mask: See above 
            memory_key_padding_mask: See above
        
        Returns:
            Tensor: (N, T, num_tokens) logits for the target sequence
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        logits = self.generator(outs)

        # Compute loss
        # Forward is only called during training/validation, so this is fine
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        # I'm sketchy about this. also C = vocab_size now
        targets = trg.view(B*T)
        # print(logits, "logits")
        # print(targets, "targets")
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def encode(self, src, src_mask):
        src_pos_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(src_pos_emb, src_mask)

    def decode(self, tgt, memory, tgt_mask):
        tgt_pos_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(tgt_pos_emb, memory, tgt_mask)