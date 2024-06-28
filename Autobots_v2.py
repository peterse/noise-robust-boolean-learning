"""
Code based on Nanogpt:

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class AutobotConfig:
    context_size: int = 32
    vocab_size: int = 2 # round to multiple of 2
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.0
    d_ff: int = 1024
    activation: str = 'gelu'
    standard_positional_encoding: bool = False
    loss_type: str = 'cross_entropy'
    bias: bool = True # True: bias in Linears and LayerNorms, False: a bit better and faster
    tie_weights: bool = True # chooses if the weights of the embedding layer and the output layer should be tied


class PositionalEncoding(nn.Module):

    """Implements the Positional Enconding as in the paper "Attention is All You Need" https://arxiv.org/abs/1706.03762
        The positional encoding is learned and added to the token embeddings. The positional encoding is a sinusoidal function.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Arguments:
            d_model: int, the number of expected features in the input
            dropout: float, the dropout value
            max_len: int, the maximum length of the input sequences
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        Returns:
            Tensor, shape ``[batch_size, seq_len, embedding_dim]``1
        """
        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)

class Autobot(nn.Module):
    """Implements the Autobot model. The model is a transformer decoder with a final linear layer to predict the next token."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_size is not None
        self.config = config

        if config.standard_positional_encoding:
            pos_emb = PositionalEncoding(config.d_model, config.dropout, config.context_size)
        else:
            pos_emb = nn.Embedding(config.context_size, config.d_model)

        if config.activation == 'gelu':
            activation = F.gelu
        elif config.activation == 'relu':
            activation = F.relu
        else:
            raise ValueError(f'Invalid activation: {config.activation}')

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.d_model, nhead=config.n_head, dim_feedforward=config.d_ff, dropout=config.dropout, 
                                                   activation=activation, batch_first=True, bias=config.bias)

        self.transformer = nn.ModuleDict(dict(
            embedding = nn.Embedding(config.vocab_size, config.d_model),
            positional_encoding = pos_emb,
            drop = nn.Dropout(config.dropout),
            decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=config.n_layer),
            normalization = nn.LayerNorm(config.d_model, bias=config.bias),
        ))
        
        self.linear_output_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.d_model = config.d_model

        # This is good if we are working with the embedded positional encoding
        if config.tie_weights:
            self.transformer.embedding.weight = self.linear_output_layer.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.standard_positional_encoding:
            n_params -= self.transformer.positional_encoding.weight.numel()
        return n_params

    # TODO: allow for custom initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_seq, targets=None):
        """Arguments:
                token_seq: Tensor, shape [batch_size, sequence_length]
                targets: Tensor, shape [batch_size, sequence_length]
            Returns:
                logits: Tensor, shape [batch_size, sequence_length, vocab_size]
                loss: Tensor, shape []
        """

        device = token_seq.device
        batch_size, sequence_lenght = token_seq.size()
        assert sequence_lenght <= self.config.context_size, f"Cannot forward sequence of length {sequence_lenght}, block size is only {self.config.context_size}"
        pos = torch.arange(0, sequence_lenght, dtype=torch.long, device=device) # shape (sequence_lenght,)
        

        tok_emb = self.transformer.embedding(token_seq) # token embeddings of shape (batch_size, sequence_lenght, d_model)

        if self.config.standard_positional_encoding:
            x = self.transformer.positional_encoding(tok_emb)
        else:
            pos_emb = self.transformer.positional_encoding(pos) # position embeddings of shape (sequence_lenght, d_model)
            x = self.transformer.drop(tok_emb + pos_emb)
            
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sequence_lenght)
        blank_mem = torch.zeros((batch_size, sequence_lenght, self.d_model), device=device)
        
        x = self.transformer.decoder(x, blank_mem, tgt_mask=tgt_mask, tgt_is_causal=True)
        x = self.transformer.normalization(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.linear_output_layer(x)
            if self.config.loss_type == 'cross_entropy':
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            elif self.config.loss_type == 'label_smoothed_cross_entropy':
                loss = F.kl_div(F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1), F.softmax(targets.view(-1), dim=-1), reduction='batchmean')
            # TODO: add more loss types
            else:
                raise ValueError(f'Invalid loss type: {self.config.loss_type}')
        else:
            # inference-time mini-optimization: only forward the linear_output_layer on the very last position
            logits = self.linear_output_layer(x[:, [-1], :]) # note: using list [-1] to preserve the time dimension
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Function for configuring the optimizer. It uses AdamW with a linear warmup and decay schedule.
            Arguments:
                weight_decay: float, the weight decay
                learning_rate: float, the learning rate
                betas: tuple, the beta parameters for Adam
                device_type: str, the device type
            Returns:
                optimizer: AdamW, the optimizer
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, token_seq, max_new_tokens=1):
        """
        Generate new tokens given an input sequence. Puts the model in eval state
        Arguments:
            token_seq: Tensor, shape [batch_size, sequence_length]
            max_new_tokens: int, maximum number of tokens to generate
        Returns:
            Tensor, shape [batch_size, sequence_length + max_new_tokens]
        """
        self.eval()
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at context_size
            token_seq_cond = token_seq if token_seq.size(1) <= self.config.context_size else token_seq[:, -self.config.context_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(token_seq_cond)
            token_seq_next = torch.argmax(logits, dim=-1)
            # append sampled index to the running sequence and continue
            token_seq = torch.cat((token_seq, token_seq_next), dim=1)

        return token_seq
    

    

