"""decoder_transformer.py - transformer model (decoder-only or encoder-only) for forecasting and next-bit prediction plus hyperparameter tuning."""

import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import math


from ray import train



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

class TransformerDecoder(nn.Module):
    """The model is a transformer decoder with a final linear layer to predict the next token."""

    def __init__(self, context_size,  vocab_size, n_layer, n_head, d_model, dropout, d_ff, activation, standard_positional_encoding, 
                 loss_type, bias, tie_weights, embedding, mode):
        
        """Arguments:
            context_size: int, the size of the context windown for the positional encoding
            vocab_size: int, the size of the vocabulary
            n_layer: int, the number of layers
            n_head: int, the number of heads
            d_model: int, the number of expected features in the input
            dropout: float, the dropout value
            d_ff: int, the number of features in the feedforward network
            activation: str, the activation function
            standard_positional_encoding: bool, whether to use the positional encoding from 'Attention is All you Need'
            loss_type: str, the loss function
            bias: bool, whether to use bias
            tie_weights: bool, whether to tie the weights
            embedding: str, the embedding type, 'embedding' or 'one_hot'. Notice that the later only works when d_model=vocab_size
            mode: str, 'decoder' or 'encoder'
        """
            
  
        super().__init__()
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        self.d_ff = d_ff
        self.activation = activation
        self.standard_positional_encoding = standard_positional_encoding
        self.loss_type = loss_type
        self.bias = bias
        self.tie_weights = tie_weights
        self.embedding = embedding
        self.mode = mode


        # Chooses to use the positional encoding from 'Attention is All you Need' or a learned positional encoding
        if self.standard_positional_encoding:
            pos_emb = PositionalEncoding(self.d_model, self.dropout, self.context_size)
        else:
            pos_emb = nn.Embedding(self.context_size, self.d_model)

        if self.activation == 'gelu':
            activation_function = F.gelu
        elif self.activation == 'relu':
            activation_function = F.relu
        else:
            raise ValueError(f'Invalid activation: {self.activation}')
        
        if self.mode == 'decoder':
            decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.n_head, dim_feedforward=self.d_ff, dropout=self.dropout, 
                                                        activation=activation_function, batch_first=True, bias=self.bias)
            core_encoder_or_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=self.n_layer)

        # WARNING: The name is confuse, but this seems to be the best choice for a good pipeline    
        elif self.mode == 'encoder':
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head, dim_feedforward=self.d_ff, dropout=self.dropout, 
                                                        activation=activation_function, batch_first=True, bias=self.bias)
            core_encoder_or_decoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.n_layer)


        self.transformer = nn.ModuleDict(dict(
                                embedding = nn.Embedding(self.vocab_size, self.d_model),
                                positional_encoding = pos_emb,
                                drop = nn.Dropout(self.dropout),
                                decoder = core_encoder_or_decoder,
                                normalization = nn.LayerNorm(self.d_model, bias=self.bias)))
        
        self.linear_output_layer = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # This is good if we are working with the embedded positional encoding
        if tie_weights:
            self.transformer.embedding.weight = self.linear_output_layer.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

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
                targets: Tensor, shape [batch_size, sequence_length].
            Returns:
                logits: Tensor, shape [batch_size, sequence_length, vocab_size]
                loss: Tensor, shape []
        """

        device = token_seq.device
        batch_size, sequence_lenght = token_seq.size()
        assert sequence_lenght <= self.context_size, f"Cannot forward sequence of length {sequence_lenght}, block size is only {self.context_size}"
        pos = torch.arange(0, sequence_lenght, dtype=torch.long, device=device) # shape (sequence_lenght,)
        
        if self.embedding == 'embedding':
            tok_emb = self.transformer.embedding(token_seq) # token embeddings of shape (batch_size, sequence_lenght, d_model)
        elif self.embedding == 'one_hot':
            assert self.d_model == self.vocab_size, f'One-hot encoding only works when d_model=vocab_size, got {self.d_model} and {self.vocab_size}'
            tok_emb = F.one_hot(token_seq, num_classes=self.vocab_size).float()
        else:
            raise ValueError(f'Invalid embedding type: {self.embedding}')

        if self.standard_positional_encoding:
            x = self.transformer.positional_encoding(tok_emb)
        else:
            pos_emb = self.transformer.positional_encoding(pos) # position embeddings of shape (sequence_lenght, d_model)
            x = self.transformer.drop(tok_emb + pos_emb)

        blank_mem = torch.zeros((batch_size, sequence_lenght, self.d_model), device=device)

        if self.mode == 'decoder':
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(sequence_lenght)
            x = self.transformer.decoder(x, blank_mem, tgt_mask=tgt_mask)
        elif self.mode == 'encoder':
            x = self.transformer.decoder(x)
        x = self.transformer.normalization(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.linear_output_layer(x)
            if self.loss_type == 'cross_entropy':
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            elif self.loss_type == 'label_smoothed_cross_entropy':
                loss = F.kl_div(F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1), F.softmax(targets.view(-1), dim=-1), reduction='batchmean')
            # TODO: add more loss types
            else:
                raise ValueError(f'Invalid loss type: {self.loss_type}')
        else:
            # inference-time mini-optimization: only forward the linear_output_layer on the very last position
            logits = self.linear_output_layer(x[:, [-1], :]) # note: using list [-1] to preserve the time dimension
            loss = None

        return logits, loss
    

def shift_and_append(raw_data, shift_step=1):
    
    out = []
    
    for sample in raw_data:

        X = torch.tensor(sample[:-shift_step])
        y = torch.tensor(sample[shift_step:])

        out.append([X,y])
        
    return out

    
    
def train_loop(config, data, checkpoint_dir=None):
    """Train the transformer (decoder-only or encoder-only) from within a raytune context. 
    
        Everything in this function needs to be reachable from the scope
        of a raytune process called from wherever you're calling it from.
    """
    # batch_size, epoch and iteration
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    context_size = config["context_size"]
    vocab_size = config["vocab_size"]
    n_layer = config["n_layer"]
    n_head = config["n_head"]
    d_model = config["d_model"]
    dropout = config["dropout"]
    d_ff = config["d_ff"]
    activation = config["activation"]
    standard_positional_encoding = config["standard_positional_encoding"]
    loss_type = config["loss_type"]
    bias = config["bias"]
    tie_weights = config["tie_weights"]
    embedding = config["embedding"]
    mode = config["mode"]
    device = config["device"]


    # Data setup: WE have a fixed train/val split of 80/20
    n_train = int(len(data) * 0.8)
    data = torch.tensor(data)
    source_target_sequences = shift_and_append(data, shift_step=1)
    data_loader = DataLoader(source_target_sequences[:n_train], batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(source_target_sequences[n_train:], batch_size=batch_size, shuffle=True)
    seq_len = data.shape[1]


    model = TransformerDecoder(context_size,  vocab_size, n_layer, n_head, d_model, dropout, d_ff, activation, standard_positional_encoding, 
                              loss_type, bias, tie_weights, embedding, mode)
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(epochs):  # You can adjust the number of epochs
        model.train()
        total_train_loss = 0
        for batch in data_loader:
        
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            logits, loss = model(X, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        total_train_loss += loss.detach().item()

        total_val_loss = 0
        model.eval()
    
        with torch.no_grad():
            for batch in val_data_loader:
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                logits, loss = model(X, y)
                total_val_loss += loss.detach().item()
        

        train.report({"loss": (total_val_loss / len(val_data_loader))})
