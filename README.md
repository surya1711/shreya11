# shreya11


1 ) Fake News Detection with NLP and Deep Learning

"""

# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Load datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Add labels to the datasets
fake_df['label'] = 0  # 0 for fake news
true_df['label'] = 1  # 1 for true news

# Combine the datasets
df = pd.concat([fake_df, true_df], axis=0)
df = df.reset_index(drop=True)

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(df['label'].value_counts())

# Data visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df)
plt.title('Distribution of Fake vs Real News')
plt.xlabel('Label (0: Fake, 1: Real)')
plt.ylabel('Count')
plt.show()

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, HTML tags, special characters, numbers
    text = re.sub(r'\[.*?\]|\W|https?://\S+|www\.\S+|<.*?>+|\n|\w*\d\w*', ' ', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into text
    text = ' '.join(tokens)
    
    return text

# Apply preprocessing to the text column
print("Preprocessing text data...")
df['processed_text'] = df['text'].apply(preprocess_text)

# Prepare features and target
X = df['processed_text']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert text to TF-IDF features
print("Converting text to TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Function to train and evaluate multiple models
def train_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Bernoulli Naive Bayes': BernoulliNB()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        # Print results
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    return results

# Train and evaluate traditional ML models
print("Training and evaluating traditional ML models...")
results = train_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test)

# Visualize model comparison
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values())
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Get the best performing model
best_model_name = max(results, key=results.get)
print(f"Best performing model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")

# Deep Learning Model
# Convert to dense arrays for neural network
X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()

# Build neural network model
def build_nn_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

# Train neural network model
print("Training neural network model...")
nn_model = build_nn_model(X_train_dense.shape[1])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train the model
history = nn_model.fit(
    X_train_dense,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the neural network model
nn_loss, nn_accuracy = nn_model.evaluate(X_test_dense, y_test)
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()

# Make predictions with neural network
y_pred_nn = (nn_model.predict(X_test_dense) > 0.5).astype("int32")

# Print classification report for neural network
print("Neural Network Classification Report:")
print(classification_report(y_test, y_pred_nn))

# Plot confusion matrix for neural network
cm_nn = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Neural Network')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance analysis for the best traditional model
if best_model_name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Decision Tree']:
    best_model = None
    
    if best_model_name == 'Logistic Regression':
        best_model = LogisticRegression(max_iter=1000, C=1.0)
    elif best_model_name == 'Random Forest':
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif best_model_name == 'Gradient Boosting':
        best_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif best_model_name == 'Decision Tree':
        best_model = DecisionTreeClassifier(random_state=42)
    
    best_model.fit(X_train_tfidf, y_train)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    if best_model_name == 'Logistic Regression':
        # For Logistic Regression
        coefficients = best_model.coef_[0]
        top_positive_coefficients = np.argsort(coefficients)[-20:]
        top_negative_coefficients = np.argsort(coefficients)[:20]
        
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.barh([feature_names[i] for i in top_positive_coefficients], coefficients[top_positive_coefficients])
        plt.title('Top Features Indicating Real News')
        plt.xlabel('Coefficient Value')
        
        plt.subplot(1, 2, 2)
        plt.barh([feature_names[i] for i in top_negative_coefficients], coefficients[top_negative_coefficients])
        plt.title('Top Features Indicating Fake News')
        plt.xlabel('Coefficient Value')
        plt.tight_layout()
        plt.show()
    
    elif best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
        # For tree-based models
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-20:]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.show()

# Save the best model
import pickle

# Save the vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save the best traditional model
if best_model_name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Decision Tree', 'Multinomial Naive Bayes', 'Bernoulli Naive Bayes']:
    best_model = None
    
    if best_model_name == 'Logistic Regression':
        best_model = LogisticRegression(max_iter=1000, C=1.0)
    elif best_model_name == 'Random Forest':
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif best_model_name == 'Gradient Boosting':
        best_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif best_model_name == 'Decision Tree':
        best_model = DecisionTreeClassifier(random_state=42)
    elif best_model_name == 'Multinomial Naive Bayes':
        best_model = MultinomialNB()
    elif best_model_name == 'Bernoulli Naive Bayes':
        best_model = BernoulliNB()
    
    best_model.fit(X_train_tfidf, y_train)
    
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"Best model ({best_model_name}) saved as 'best_model.pkl'")

# Save neural network model if it performs better
if nn_accuracy > results[best_model_name]:
    nn_model.save('nn_model.h5')
    print("Neural network model saved as 'nn_model.h5'")

# Function to predict on new text
def predict_news(text, model_type='traditional'):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Vectorize
    text_tfidf = vectorizer.transform([processed_text])
    
    if model_type == 'traditional':
        # Load the best traditional model
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Make prediction
        prediction = model.predict(text_tfidf)[0]
    else:
        # Load neural network model
        nn_model = tf.keras.models.load_model('nn_model.h5')
        
        # Convert to dense array
        text_dense = text_tfidf.toarray()
        
        # Make prediction
        prediction = (nn_model.predict(text_dense) > 0.5).astype("int32")[0][0]
    
    return "Real" if prediction == 1 else "Fake"

# Example usage
print("\nExample prediction:")
example_text = "Scientists have discovered a new treatment that can cure all types of cancer with just one pill."
print(f"Text: {example_text}")
print(f"Prediction: {predict_news(example_text)}")


"""

**************************************************************************************************************************************************
Additional Modifications and Improvements
Here are some additional modifications that could enhance the fake news detection system:

1. Cross-Validation for More Robust Evaluation
'''

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
def cross_validate_models(X, y):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Bernoulli Naive Bayes': BernoulliNB()
    }
    
    cv_results = {}
    
    for name, model in models.items():
        print(f"Cross-validating {name}...")
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_results[name] = cv_scores.mean()
        print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return cv_results

# Run cross-validation
cv_results = cross_validate_models(X_train_tfidf, y_train)

'''
###################################################################

2. Hyperparameter Tuning

'''
from sklearn.model_selection import GridSearchCV

# Example for Logistic Regression
def tune_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Tune Logistic Regression
best_lr = tune_logistic_regression(X_train_tfidf, y_train)

'''
###################################################################
3. Ensemble Methods
'''
from sklearn.ensemble import VotingClassifier

def create_ensemble(X_train, y_train):
    # Create base classifiers
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    nb = MultinomialNB()
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr),
            ('rf', rf),
            ('gb', gb),
            ('nb', nb)
        ],
        voting='soft'  # Use predicted probabilities
    )
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    return ensemble

# Create and evaluate ensemble
ensemble_model = create_ensemble(X_train_tfidf, y_train)
ensemble_accuracy = ensemble_model.score(X_test_tfidf, y_test)
print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
'''
###################################################################

4. Advanced Text Features
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

def create_advanced_features(X_train, X_test):
    # Word-level TF-IDF
    word_tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    # Character-level TF-IDF
    char_tfidf = TfidfVectorizer(
        analyzer='char',
        max_features=3000,
        ngram_range=(3, 6),
        sublinear_tf=True
    )
    
    # Count vectors for specific patterns
    count_vec = CountVectorizer(
        max_features=1000,
        binary=True,
        ngram_range=(1, 1)
    )
    
    # Combine features
    features = FeatureUnion([
        ('word_tfidf', word_tfidf),
        ('char_tfidf', char_tfidf),
        ('count_vec', count_vec)
    ])
    
    # Transform data
    X_train_features = features.fit_transform(X_train)
    X_test_features = features.transform(X_test)
    
    return X_train_features, X_test_features, features

# Create advanced features
X_train_advanced, X_test_advanced, feature_union = create_advanced_features(X_train, X_test)
'''
###################################################################
5. Bidirectional LSTM Model for Better Text Understanding

'''

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM

def build_lstm_model(X_train, X_test, y_train, y_test):
    # Tokenize text
    max_words = 10000
    max_len = 200
    
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
    
    # Build model
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_pad,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f"LSTM Model Accuracy: {accuracy:.4f}")
    
    return model, tokenizer, history

# Build and train LSTM model
lstm_model, tokenizer, lstm_history = build_lstm_model(X_train, X_test, y_train, y_test)

'''
###################################################################

6. Web Application for Deployment
'''

from flask import Flask, request, render_template, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the model and vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, HTML tags, special characters, numbers
    text = re.sub(r'\[.*?\]|\W|https?://\S+|www\.\S+|<.*?>+|\n|\w*\d\w*', ' ', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into text
    text = ' '.join(tokens)
    
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        
        # Preprocess the text
        processed_text = preprocess_text(news_text)
        
        # Vectorize
        text_tfidf = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        result = "Real" if prediction == 1 else "Fake"
        
        return render_template('index.html', prediction_text=f'The news is {result}', news_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)
'''
###################################################################

7. Real-time Monitoring and Model Updating

'''
def update_model_with_new_data(new_data_file, current_model_file, current_vectorizer_file):
    # Load current model and vectorizer
    with open(current_model_file, 'rb') as f:
        current_model = pickle.load(f)
    
    with open(current_vectorizer_file, 'rb') as f:
        current_vectorizer = pickle.load(f)
    
    # Load new data
    new_data = pd.read_csv(new_data_file)
    
    # Preprocess new data
    new_data['processed_text'] = new_data['text'].apply(preprocess_text)
    
    # Extract features and labels
    X_new = new_data['processed_text']
    y_new = new_data['label']
    
    # Transform new data using current vectorizer
    X_new_tfidf = current_vectorizer.transform(X_new)
    
    # Update model with new data (partial_fit for incremental learning)
    if hasattr(current_model, 'partial_fit'):
        current_model.partial_fit(X_new_tfidf, y_new)
    else:
        # For models that don't support partial_fit, retrain on combined data
        # This would require storing the original training data or using a different approach
        pass
    
    # Save updated model
    with open('updated_model.pkl', 'wb') as f:
        pickle.dump(current_model, f)
    
    print("Model updated with new data and saved as 'updated_model.pkl'")
'''
***************************************************END**************************************************
