import pickle
import torch
from nltk.tokenize import word_tokenize

from rnn_model import SimpleRNN


class RNNService:

    def __init__(self):

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load vocabulary
        with open("model_artifacts/rnn_vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)

        # Create model
        self.model = SimpleRNN(len(self.vocab))

        # Load trained weights
        self.model.load_state_dict(
            torch.load(
                "model_artifacts/rnn_model.pth",
                map_location=self.device
            )
        )

        # Move model to device
        self.model.to(self.device)

        # Evaluation mode
        self.model.eval()

        print("RNN loaded successfully")
        print("RNN vocabulary size:", len(self.vocab))
        print("RNN device:", self.device)

    def text_to_token(self, text):

        # Tokenize
        tokens = word_tokenize(text.lower())

        indices = []

        for word in tokens:
            if word in self.vocab:
                indices.append(self.vocab[word])
            else:
                indices.append(self.vocab["<UNK>"])

        return indices

    def predict_answer(self, ques, threshold=0.5):

        # Convert question to numerical tokens
        num_ques = self.text_to_token(ques)

        # Convert to tensor
        ques_tensor = torch.tensor(
            num_ques,
            dtype=torch.long
        ).unsqueeze(0)

        # Move input to same device as model
        ques_tensor = ques_tensor.to(self.device)

        # Prediction
        with torch.no_grad():
            output = self.model(ques_tensor)

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(
            output,
            dim=1
        )

        # Get highest probability
        val, ind = torch.max(probs, dim=1)

        # Confidence check
        if val.item() < threshold:
            return "Not exactly sure"

        # Convert tensor to Python integer
        ind = ind.item()

        # Convert index back to word
        for word, index in self.vocab.items():
            if index == ind:
                return word

        return "<UNK>"