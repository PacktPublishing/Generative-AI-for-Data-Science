
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_20newsgroups
import pickle


# Load the newsgroup dataset
data = fetch_20newsgroups()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.2)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Create a Random Forest classifier
classifier = RandomForestClassifier()

# Fit the classifier to the training data
classifier.fit(X_train_tfidf, y_train)

# Predict the labels of the test data
y_pred = classifier.predict(X_test_tfidf)

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average='macro')
print('F1 score:', f1)

pickle.dump({'X_test': X_test, 
             'y_test': y_test, 
             'f1_score_rf': f1}, open("saved.pk", "wb"))
