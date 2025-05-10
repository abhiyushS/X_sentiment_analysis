# Twitter Sentiment Analysis using DistilBERT

This project performs sentiment analysis on tweets using a fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased) model. The goal is to classify tweets as **positive**, **negative**, or **neutral** based on their content.

---

## 🚀 Features

- Preprocessing of tweet text (cleaning, tokenization)
- Fine-tuning of DistilBERT on labeled tweet sentiment dataset
- Model evaluation using accuracy and classification metrics
- Inference pipeline for predicting sentiment on new tweets

---

## 📁 Project Structure

twiter sentiment analysis/
│
├── data/ # Dataset files (train/test CSVs or JSONs)
├── notebooks/ # Jupyter notebooks (EDA, training, etc.)
├── src/ # Source code (preprocessing, training, inference)
├── my_distilbert_model/ # Fine-tuned DistilBERT model (excluded from Git)
├── results/ # Training checkpoints and logs (excluded from Git)
├── requirements.txt # Python dependencies
├── .gitignore # Ignoring model & large file folders
└── README.md # Project documentation

2. Install dependencies
pip install -r requirements.txt


3. Download or restore the model (optional)
The my_distilbert_model/ folder is excluded from GitHub due to its size. To run inference:

Option 1: Retrain the model using provided training scripts or notebooks

Option 2: Manually place the trained model folder (my_distilbert_model/) in the project root

🧠 Model Details
Base Model: distilbert-base-uncased from Hugging Face Transformers

Fine-tuned on: A labeled tweet sentiment dataset

Output: One of three classes — Positive, Negative, or Neutral


📊 Example Inference
python
Copy
Edit
from transformers import pipeline

# Load fine-tuned model
classifier = pipeline(
    "text-classification",
    model="./my_distilbert_model",
    tokenizer="./my_distilbert_model"
)

text = "I love the new design of the app!"
result = classifier(text)
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.98}]


📚 License
This project is licensed under the MIT License. See the LICENSE file for details.

✍️ Author
Abhiyush S
Feel free to contribute or open issues for suggestions or improvements.


