import torch
from torch.utils.data import Dataset


class MusicDataset(Dataset):
    def __init__(self, text, seq_length=64):
        tokens = text.split()

        vocab = set(tokens)
        # stoi = string to index
        self.stoi = {word: i for i, word in enumerate(vocab)}
        # itos = index to string
        self.itos = {i: word for word, i in self.stoi.items()}

        # we convert the entire text into a list of indices
        self.data = [self.stoi[token] for token in tokens]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # input sequence
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        # target sequence (next tokens)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y
