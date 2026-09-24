"use client";

import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [suggestion, setSuggestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loadingSuggestion, setLoadingSuggestion] = useState(false);
  const [loadingAnswer, setLoadingAnswer] = useState(false);

  // Ask LSTM for the next word
  const getNextWord = async () => {
    if (!question.trim()) return;

    setLoadingSuggestion(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict-next-word",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: question,
          }),
        }
      );

      const data = await response.json();

      setSuggestion(data.word);
    } catch (error) {
      console.error("LSTM error:", error);
    } finally {
      setLoadingSuggestion(false);
    }
  };

  // Send question through LSTM → RNN pipeline
  const askQuestion = async () => {
  if (!question.trim()) return;

  setLoadingAnswer(true);
  setAnswer("");

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/ask",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
        }),
      }
    );

    const data = await response.json();

    setAnswer(data.answer);
  } catch (error) {
    console.error("RNN error:", error);
  } finally {
    setLoadingAnswer(false);
  }
};

  // Accept LSTM suggestion
  const acceptSuggestion = () => {
    setQuestion(
      question.trim() + " " + suggestion
    );

    setSuggestion("");
  };

  return (
    <main className="min-h-screen bg-gray-100 flex items-center justify-center p-6">

      <div className="w-full max-w-3xl bg-white rounded-2xl shadow-lg p-8">

        {/* Heading */}
        <h1 className="text-3xl font-bold text-center mb-2">
          Mini Search Engine
        </h1>

        <p className="text-center text-gray-500 mb-8">
          Complete your question with LSTM and get the answer from RNN
        </p>


        {/* Question */}
        <label className="block font-semibold mb-2">
          Your Question
        </label>

        <textarea
          value={question}
          onChange={(e) => {
            setQuestion(e.target.value);
            setAnswer("");
            setSuggestion("");
          }}
          placeholder="Start typing your question..."
          className="w-full h-32 border border-gray-300 rounded-lg p-4
                     focus:outline-none focus:ring-2 focus:ring-blue-500"
        />


        {/* LSTM button */}
        <button
          onClick={getNextWord}
          disabled={loadingSuggestion}
          className="mt-4 px-5 py-2 bg-blue-600 text-white
                     rounded-lg hover:bg-blue-700
                     disabled:opacity-50"
        >
          {loadingSuggestion
            ? "Predicting..."
            : "Get Next Word"}
        </button>


        {/* LSTM suggestion */}
        {suggestion && (
          <div className="mt-5">

            <p className="text-sm text-gray-500 mb-2">
              LSTM suggested next word
            </p>

            <button
              onClick={acceptSuggestion}
              className="px-4 py-2 bg-blue-100 text-blue-700
                         rounded-lg hover:bg-blue-200"
            >
              {suggestion}
            </button>

          </div>
        )}


        {/* Ask Question */}
        <button
          onClick={askQuestion}
          disabled={loadingAnswer || !question.trim()}
          className="w-full mt-6 bg-blue-600 text-white
                     py-3 rounded-lg font-semibold
                     hover:bg-blue-700
                     disabled:opacity-50"
        >
          {loadingAnswer
            ? "Getting Answer..."
            : "Ask Question"}
        </button>


        {/* Answer */}
        {answer && (
          <div className="mt-8 p-5 bg-gray-50 rounded-lg">

            <h2 className="font-semibold text-lg mb-2">
              Answer
            </h2>

            <p className="text-gray-700">
              {answer}
            </p>

          </div>
        )}

      </div>

    </main>
  );
}