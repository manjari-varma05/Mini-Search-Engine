import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embd = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.linear = nn.Linear(150, vocab_size)

    def forward(self, x):

        embedded = self.embd(x)

        int_hs, (final_hs, final_cs) = self.lstm(embedded)

        op = self.linear(final_hs.squeeze(0))

        return op