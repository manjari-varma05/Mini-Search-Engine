# Mini Search Engine

An AI-based Search Engine that combines an **LSTM model for next-word prediction** and a **Simple RNN model for answer prediction** with a **FastAPI backend** and a **Next.js frontend**.

The LSTM helps complete the user's question, and the RNN predicts the answer to the completed question.

## Overview

The project works in two stages:

1. The LSTM predicts the next word in a partially typed question.
2. The RNN predicts the answer to the completed question.

### Example

User enters:

> What is the capital of

LSTM predicts:

> brazil

After accepting the suggestion:

> What is the capital of brazil

RNN predicts:

> brasilia

The answer is then displayed on the frontend.

---

## System Architecture

```text
                         USER
                           |
                           v
                  Next.js Frontend
                           |
                           v
                   FastAPI Backend
                    /           \
                   /             \
                  v               v
                LSTM             RNN
        Next Word Prediction   Answer Prediction
                  |               |
                  v               v
           Suggested Word    Predicted Answer
                  \               /
                   \             /
                    v           v
                  Frontend
                      |
                      v
                     USER
```

---

## Complete Workflow

```text
User enters question
        |
        v
Next.js Frontend
        |
        | POST /predict-next-word
        v
FastAPI Backend
        |
        v
LSTM Model
        |
        v
Predicted Next Word
        |
        v
User accepts suggestion
        |
        v
Completed Question
        |
        | POST /ask
        v
FastAPI Backend
        |
        v
RNN Model
        |
        v
Predicted Answer
        |
        v
Next.js Frontend
        |
        v
Answer displayed
```

---

# LSTM - Next Word Prediction

The LSTM is responsible for predicting the next word in the user's question.

### Example

Input:

```text
What is the capital of
```

Prediction:

```text
brazil
```

### LSTM Architecture

```text
Input Question
      |
      v
Tokenization
      |
      v
Vocabulary Mapping
      |
      v
Sequence Padding
      |
      v
Embedding Layer
      |
      v
LSTM Layer
      |
      v
Linear Layer
      |
      v
Next Word Prediction
```

### LSTM Configuration

- Embedding Dimension: 100
- LSTM Hidden Size: 150
- Maximum Sequence Length: 12
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 50
- Loss Function: Cross Entropy Loss

---

# RNN - Answer Prediction

The Simple RNN is responsible for predicting the answer to the completed question.

### Example

Question:

```text
What is the capital of france
```

Prediction:

```text
paris
```

### RNN Architecture

```text
Question
    |
    v
Tokenization
    |
    v
Vocabulary Mapping
    |
    v
Embedding Layer
    |
    v
RNN Layer
    |
    v
Linear Layer
    |
    v
Answer Prediction
```

### RNN Configuration

- Embedding Dimension: 50
- RNN Hidden Size: 64
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 20
- Loss Function: Cross Entropy Loss

---

# Frontend

The frontend is built using:

- Next.js
- React
- TypeScript
- Tailwind CSS

The frontend allows the user to:

- Enter a question
- Get a next-word prediction
- Accept the predicted word
- Complete the question
- Ask the question
- View the predicted answer

### Frontend Flow

```text
Enter Question
      |
      v
Get Next Word
      |
      v
Accept Suggestion
      |
      v
Completed Question
      |
      v
Ask Question
      |
      v
Display Answer
```

---

# Backend

The backend is built using:

- Python
- FastAPI
- Uvicorn
- Pydantic

The backend loads both trained models and provides APIs for the frontend.

## API Endpoints

### Predict Next Word

**POST** `/predict-next-word`

Request:

```json
{
  "text": "What is the capital of"
}
```

Response:

```json
{
  "word": "brazil"
}
```

### Ask Question

**POST** `/ask`

Request:

```json
{
  "question": "What is the capital of brazil"
}
```

Response:

```json
{
  "answer": "brasilia"
}
```

### API Status

**GET** `/`

Response:

```json
{
  "message": "Course AI Assistant API is running"
}
```

---

# Data Preprocessing

The input text is preprocessed before being passed to the models.

The main steps are:

1. Convert text to lowercase
2. Tokenize the text
3. Convert words into numerical indices
4. Handle unknown words
5. Convert the sequence into PyTorch tensors
6. Pass the processed input to the model

For the LSTM, sequences are padded or truncated to a maximum length of 12.

---

# Model Artifacts

The trained models are stored in the backend.

### LSTM

```text
lstm_model.pth
lstm_vocab.pkl
lstm_config.pkl
```

- `lstm_model.pth` - Trained LSTM model weights
- `lstm_vocab.pkl` - LSTM vocabulary
- `lstm_config.pkl` - LSTM configuration

### RNN

```text
rnn_model.pth
rnn_vocab.pkl
```

- `rnn_model.pth` - Trained RNN model weights
- `rnn_vocab.pkl` - RNN vocabulary

---

# Project Structure

```text
RNN-LSTM/
│
├── backend/
│   ├── main.py
│   ├── lstm_model.py
│   ├── lstm_service.py
│   ├── rnn_model.py
│   ├── rnn_service.py
│   ├── requirements.txt
│   │
│   └── model_artifacts/
│       ├── lstm_model.pth
│       ├── lstm_vocab.pkl
│       ├── lstm_config.pkl
│       ├── rnn_model.pth
│       └── rnn_vocab.pkl
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── public/
│   ├── package.json
│   ├── package-lock.json
│   └── ...
│
├── .gitignore
└── README.md
```

---

# Technologies Used

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic

### Machine Learning

- PyTorch
- LSTM
- Simple RNN
- NLTK

---

# Running the Project Locally

## Backend

Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Frontend

Open another terminal and navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the Next.js development server:

```bash
npm run dev
```

The frontend will run at:

```text
http://localhost:3000
```

---

# Example Interaction

```text
User:
What is the capital of

        ↓

LSTM:
brazil

        ↓

User accepts suggestion

        ↓

Completed Question:
What is the capital of brazil

        ↓

RNN:
brasilia

        ↓

Frontend:
Answer: brasilia
```

---

# Key Features

- LSTM-based next-word prediction
- RNN-based answer prediction
- Interactive question completion
- Next.js frontend
- FastAPI backend
- PyTorch model integration
- REST API communication
- Separate LSTM and RNN services
- Real-time model inference

---

# Future Improvements

- Multi-word question completion
- Top-k next-word suggestions
- Prediction confidence scores
- Larger training dataset
- Better answer generation
- Conversational memory
- Conversational answer generation
- Improved frontend UI
- Authentication
- Deployment of frontend and backend

---

# Author

**Manjari M Varma**
