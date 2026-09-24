from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lstm_service import LSTMService
from rnn_service import RNNService


app = FastAPI(title="Course AI Assistant")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


lstm_service = LSTMService()
rnn_service = RNNService()


class TextRequest(BaseModel):
    text: str


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Course AI Assistant API is running"
    }


@app.post("/predict-next-word")
def predict_next_word(request: TextRequest):

    word = lstm_service.predict_next_word(
        request.text
    )

    return {
        "word": word
    }

@app.post("/complete-and-answer")
def complete_and_answer(request: TextRequest):

    # Step 1: LSTM predicts the next word
    predicted_word = lstm_service.predict_next_word(
        request.text
    )

    # Step 2: Build the completed question
    completed_question = request.text + " " + predicted_word

    # Step 3: Send completed question to RNN
    answer = rnn_service.predict_answer(
        completed_question
    )

    return {
        "predicted_word": predicted_word,
        "completed_question": completed_question,
        "answer": answer
    }
@app.post("/ask")
def ask_question(request: QuestionRequest):

    answer = rnn_service.predict_answer(
        request.question
    )

    return {
        "answer": answer
    }