import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)

# Load data
df_fake = pd.read_csv("https://www.kaggle.com/code/therealsampat/fake-news-detection?select=Fake.csv")
df_true = pd.read_csv("https://www.kaggle.com/code/therealsampat/fake-news-detection?select=True.csv")

# Add class labels
df_fake["class"] = 0
df_true["class"] = 1

# Combine the datasets
df_merge = pd.concat([df_fake, df_true], axis=0)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X_tfidf = tfidf_vectorizer.fit_transform(df_merge['text'])
y = df_merge['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.25, random_state=42)

# Initialize classifiers
naive_bayes = MultinomialNB()
decision_tree = DecisionTreeClassifier(random_state=42)
logistic_regression = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(random_state=42)

# Train classifiers
naive_bayes.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
logistic_regression.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the news article input from the form
        news_article = request.form['news_article']
        
        # Transform the input using the same TF-IDF vectorizer
        test_vector = tfidf_vectorizer.transform([news_article])

        # Get predictions from the models
        pred_nb = naive_bayes.predict(test_vector)
        pred_dt = decision_tree.predict(test_vector)
        pred_lr = logistic_regression.predict(test_vector)
        pred_rfc = random_forest.predict(test_vector)

        # Prepare a dictionary with the predictions
        predictions = {
            "Naive Bayes": "Fake News" if pred_nb[0] == 0 else "Not Fake News",
            "Decision Tree": "Fake News" if pred_dt[0] == 0 else "Not Fake News",
            "Logistic Regression": "Fake News" if pred_lr[0] == 0 else "Not Fake News",
            "Random Forest": "Fake News" if pred_rfc[0] == 0 else "Not Fake News"
        }

        # Render the result page with the predictions
        return render_template('result.html', article=news_article, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
