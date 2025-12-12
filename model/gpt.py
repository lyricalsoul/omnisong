import torch
from torch import nn


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=4, seq_length=64, max_len=1024, dropout=0.54,
                 gradient_checkpointing=False):
        super().__init__()
        # set embedding layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        self.dropout = nn.Dropout(dropout)  # dropout to avoid overfitting

        self.gradient_checkpointing = gradient_checkpointing

        # transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            activation='gelu',
            dropout=dropout,
            batch_first=True
        )

        # stacking multiple layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final layer norm
        self.ln_f = nn.LayerNorm(embed_size)

        # hidden state to vocab size
        self.fc_out = nn.Linear(embed_size, vocab_size)

        self.fc_out.weight = self.embedding.weight

        self.seq_length = seq_length
        self.max_len = max_len

    def forward(self, x):
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, seq_length).to(x.device)

        # avoids seeing future tokens and just freezing generation
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(x.device)

        x = self.embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        if self.gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.transformer,
                x,
                causal_mask,
                use_reentrant=False
            )
        else:
            x = self.transformer(x, mask=causal_mask)

        x = self.ln_f(x)
        logits = self.fc_out(x)

        return logits
