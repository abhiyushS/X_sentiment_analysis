#!/usr/bin/env python
# coding: utf-8

# In[50]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[51]:


import sys
get_ipython().system('{sys.executable} -m pip uninstall transformers accelerate torch datasets -y')
get_ipython().system('{sys.executable} -m pip install transformers==4.38.2 accelerate==0.28.0 torch==2.2.1 datasets==2.18.0')


# In[1]:


import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('Twitter_Data.csv')
df['category'] = df['category'] + 1  # Shift labels from -1,0,1 to 0,1,2
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# In[4]:


tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize(batch):
    texts = [str(t) if t is not None else "" for t in batch['clean_text']]  # Changed to 'clean_text'
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("category", "labels")
test_dataset = test_dataset.rename_column("category", "labels")

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])


# In[8]:


model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=3, 
    problem_type="single_label_classification"  # Fix for single-label task
)


# In[9]:


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)


# In[10]:


train_subset = train_dataset.select(range(10000))
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=test_dataset,
)
trainer.train()


# In[11]:


eval_results = trainer.evaluate()
print(eval_results)


# In[12]:


trainer.save_model("./my_distilbert_model")
tokenizer.save_pretrained("./my_distilbert_model")


# In[14]:


import torch

def predict_sentiment(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # Move inputs to MPS device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Ensure model is on the same device
    model.to(device)
    # Get prediction
    outputs = model(**inputs)
    pred = outputs.logits.argmax(-1).item()
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[pred]

while True:
    tweet = input("Enter a tweet (or 'quit' to stop): ")
    if tweet.lower() == 'quit':
        break
    result = predict_sentiment(tweet)
    print(f"Sentiment: {result}")


# In[ ]:




