import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden: int = 128, num_layers: int = 1, num_classes: int = 2, dropout: float = 0.3, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)                              # [B, T, E]
        lengths = attention_mask.sum(dim=1) if attention_mask is not None else (input_ids != 0).sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # concat last hidden states from both directions
        # h_n shape: [num_layers*2, B, H] -> take last layer (both directions)
        h_fwd = h_n[-2,:,:]
        h_bwd = h_n[-1,:,:]
        h = torch.cat([h_fwd, h_bwd], dim=1)                       # [B, 2H]
        h = self.dropout(h)
        logits = self.fc(h)                                        # [B, C]
        return logits
