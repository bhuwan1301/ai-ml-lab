import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Load the dataset
try:
    df = pd.read_csv('mhqa-b.csv')
except FileNotFoundError:
    print("Error: 'mhqa-b.csv' file not found. Please make sure the file is in the same directory.")
    exit()

# 2. Select suitable parameters
# We use the text of the 'question' to predict the 'topic'
X = df['question']
y = df['topic']

# 3. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Convert text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train the Classifier (Logistic Regression)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_tfidf, y_train)

# 6. Make Predictions
y_pred = clf.predict(X_test_tfidf)

# 7. Create Confusion Matrix
labels = clf.classes_
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# 8. Print the Confusion Matrix "acche se"
print("\n" + "="*80)
print("                       CLASSIFICATION RESULTS                       ")
print("="*80)
print(f"Target Class: 'topic' | Input Feature: 'question'")
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("-" * 80)
print("CONFUSION MATRIX")
print("(Rows = Actual Values, Columns = Predicted Values)")
print("-" * 80)
# Adjust pandas display options to ensure columns don't get hidden
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(cm_df)
print("="*80 + "\n")