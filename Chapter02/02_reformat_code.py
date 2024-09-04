import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the 20 Newsgroups dataset (testing purposes only)
twenty_train = fetch_20newsgroups(subset='train')
twenty_test = fetch_20newsgroups(subset='test')

# Convert text data into a pandas DataFrame
data = pd.DataFrame({'text': twenty_test.data, 'labels': twenty_test.target})

# Create a tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the data
data['tokens'] = data['text'].apply(tokenizer.tokenize)

# Create a model
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Train the model (for testing purposes only)
# Note: This is just for demonstration, you should not train models on test data in practice!
model.fit(data['tokens'], data['labels'])

# Save the model (for testing purposes only)
model.save_pretrained('test_model.pt')