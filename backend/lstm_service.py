import pickle
import torch
from nltk.tokenize import word_tokenize

from lstm_model import LSTM


class LSTMService:

    def __init__(self):

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load vocabulary
        with open("model_artifacts/lstm_vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)

        # Load config
        with open("model_artifacts/lstm_config.pkl", "rb") as f:
            config = pickle.load(f)

        self.max_len = config["max_len"]

        # Create model with same vocabulary size
        self.model = LSTM(len(self.vocab))

        # Load trained weights
        self.model.load_state_dict(
            torch.load(
                "model_artifacts/lstm_model.pth",
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

        print("LSTM loaded successfully")
        print("Vocabulary size:", len(self.vocab))
        print("Max length:", self.max_len)
        print("Device:", self.device)
    def predict_next_word(self, ques):

    # Tokenize
      tokenized_ques = word_tokenize(ques.lower())

      # Numericalize
      num_ques = []

      for token in tokenized_ques:
          if token in self.vocab:
              num_ques.append(self.vocab[token])
          else:
              num_ques.append(self.vocab["<unk>"])

      # Handle longer questions
      if len(num_ques) > self.max_len:
          num_ques = num_ques[-self.max_len:]

      # Padding
      padded_ques = torch.tensor(
          [0] * (self.max_len - len(num_ques)) + num_ques,
          dtype=torch.long
      ).unsqueeze(0)

      # Move input to same device as model
      padded_ques = padded_ques.to(self.device)

      # Prediction
      with torch.no_grad():
          output = self.model(padded_ques)

      # Get predicted word index
      _, ind = torch.max(output, dim=1)

      ind = ind.item()

      # Convert index back to word
      for word, index in self.vocab.items():
          if index == ind:
              return word

      return "<unk>"