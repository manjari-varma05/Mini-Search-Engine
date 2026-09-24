import torch.nn as nn


class SimpleRNN(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.emb = nn.Embedding(
            vocab_size,
            embedding_dim=50
        )

        self.rnn = nn.RNN(
            input_size=50,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(
            in_features=64,
            out_features=vocab_size
        )

    def forward(self, x):

        # Convert word indices to embeddings
        x = self.emb(x)

        # Pass embeddings through RNN
        x, h = self.rnn(x)

        # Take final hidden state
        h = h.squeeze(0)

        # Predict over vocabulary
        output = self.fc(h)

        return output