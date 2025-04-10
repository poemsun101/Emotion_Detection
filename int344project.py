

"A Comprehensive Approach to Sentiment Analysis: Combining Logistic Regression and Naive Bayes for Enhanced Text Classification"**
"""

import pandas as pd

# Load the IMDb dataset
df = pd.read_csv('/content/IMDB Dataset.csv')
df.head()

df.shape

df.isnull().sum()

df.info()

df.describe()

from matplotlib import pyplot as plt
import seaborn as sns
df.groupby('sentiment').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

"""**Data Preprocessing**"""

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_word = set(stopwords.words('english'))
stop_words = list(stop_word)
print(stop_words[:15])

import re
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
   #lowercase convert
    text = text.lower()

    # Remove stopwords and apply lemmatization
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Rejoin tokens into a single string
    return ' '.join(tokens)

# Apply the preprocessing function to the 'text' column
df['cleaned_reviews'] = df['review'].apply(preprocess_text)

"""**Feature Extraction (Vectorization)**"""

from sklearn.feature_extraction.text import TfidfVectorizer
#Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit transform the cleaned text into vectors
X = vectorizer.fit_transform(df['cleaned_reviews']).toarray()

#target labels (sentiment)
y = df['sentiment']  #'sentiment' column has the labels

from sklearn.model_selection import train_test_split

#Split the data(80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""**Apply Logistic Regression and Naive Bayes Models**"""

from sklearn.linear_model import LogisticRegression

# Initialize and train the model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_logreg = log_reg.predict(X_test)

from sklearn.naive_bayes import MultinomialNB

# Initialize and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_nb = nb_model.predict(X_test)

"""**Evaluate the Models**"""

from sklearn.metrics import accuracy_score, classification_report
# Logistic Regression
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))

#Naive Bayes
print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

import matplotlib.pyplot as plt

# Accuracy values for the models
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Data for plotting
models = ['Logistic Regression', 'Naive Bayes']
accuracies = [accuracy_logreg, accuracy_nb]

# Create the bar plot
plt.bar(models, accuracies, color=['cyan', 'purple'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()

"""**Combine Predictions(optional)**"""

import numpy as np

# cre dictionary to map string labels to numerical values
label_mapping = {'negative': 0, 'positive': 1}

# con string labels to numerical values
y_pred_logreg_num = np.array([label_mapping[label] for label in y_pred_logreg])
y_pred_nb_num = np.array([label_mapping[label] for label in y_pred_nb])

# average the numerical predictions from both model
final_pred_num = np.round((y_pred_logreg_num + y_pred_nb_num) / 2)

# Con numerical predictions back to string labels
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
final_pred = np.array([inverse_label_mapping[pred] for pred in final_pred_num])
print("Combined Model Accuracy:", accuracy_score(y_test, final_pred))
# print("Combined Model Classification Report:")
# print(classification_report(y_test, final_pred))

from sklearn.metrics import classification_report
#Logistic Regression
print("Logistic Regression:")
print(classification_report(y_test, y_pred_logreg))

# Naive Bayes
print("\nNaive Bayes:")
print(classification_report(y_test, y_pred_nb))

""" **LOGISTIC REGRESSION:- precision,recall,f1-score visulization**"""

import matplotlib.pyplot as plt
import numpy as np

# precision, recall, and f1-score (Logistic Regression report)
metrics = ['Precision', 'Recall', 'F1-Score']
negative_scores = [0.89, 0.87, 0.88]
positive_scores = [0.88, 0.90, 0.89]

# Setting the positions and width for the bars
x = np.arange(len(metrics))  # Lab locations
width = 0.35  # Width of the bars

# Create the plot
fig, a = plt.subplots(figsize=(8, 5))

# Plot bars for negative and positive classes
bars1 = a.bar(x - width/2, negative_scores, width, label='Negative', color='cyan')
bars2 = a.bar(x + width/2, positive_scores, width, label='Positive', color='black')

# Add labels, title, and legend
a.set_ylabel('Scores')
a.set_title('Precision, Recall, F1-Score for Positive and Negative Classes')
a.set_xticks(x)
a.set_xticklabels(metrics)
a.legend()
plt.show()

"""**Naive Bayes:- precision,recall,f1-score visulization**"""

import matplotlib.pyplot as plt
import numpy as np

# precision, recall, and f1-score (Navies Bayes report)
metrics = ['Precision', 'Recall', 'F1-Score']
negative_scores = [0.85,0.85,0.85 ]
positive_scores = [0.85,0.86,0.85 ]

# Setting the positions and width for the bars
x = np.arange(len(metrics))  # Label locations
width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot bars for negative and positive classes
bars1 = ax.bar(x - width/2, negative_scores, width, label='Negative', color='coral')
bars2 = ax.bar(x + width/2, positive_scores, width, label='Positive', color='MediumVioletRed')

# Add labels, title, and legend
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, F1-Score for Positive and Negative Classes')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.show()

"""**(optional) daigram**"""

import matplotlib.pyplot as plt

# Data for precision, recall, and f1-score (from your Logistic Regression report)
metrics = ['Precision', 'Recall', 'F1-Score']
negative_scores = [0.89, 0.87, 0.88]
positive_scores = [0.88, 0.90, 0.89]

# Labelsss for the two classes
labels = ['Ne', 'Po']

# Drow a pie chart for each metric
fig, a = plt.subplots(1, 3, figsize=(10, 5))

for i, metric in enumerate(metrics):
    # current metric
    data = [negative_scores[i], positive_scores[i]]

    #pie chart
    a[i].pie(data, labels=labels, autopct='%1.1f%%', colors=['MediumVioletRed', 'Teal'], startangle=90)
    a[i].set_title(metric)

plt.show()

"""**DROW A CONFUSION MATRIX**"""

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion Matrix for Logistic Regression
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
sns.heatmap(cm_logreg, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt="d", cmap='Greens')
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

"""**MODEL COMPARISON PRECISION, RECALL, F1SCORE**"""

import matplotlib.pyplot as plt
import numpy as np

# Scores for precision, recall, and f1-score for each model
precision_scores = [0.88, 0.89]  # [Logistic Regression, Naive Bayes]
recall_scores = [0.90, 0.87]
f1_scores = [0.89, 0.88]

# Model names
models = ['Logistic Regression', 'Naive Bayes']

# Create positions for the bars
x = np.arange(len(models))
width = 0.2

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot Precision, Recall, and F1-Score bars for each model
ax.bar(x - width, precision_scores, width, label='Precision', color='teal')
ax.bar(x, recall_scores, width, label='Recall', color='coral')
ax.bar(x + width, f1_scores, width, label='F1-Score', color='crimson')

# Add labels, title, and x-ticks
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_title('Model Comparison: Precision, Recall, F1-Score')
ax.legend()
plt.show()

"""**precision_recall_curve**"""

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Convert 'positive'/'negative' labels to 0 and 1
lb = LabelBinarizer()
y_test_binary = lb.fit_transform(y_test)

# Assuming y_pred_proba (predicted probabilities for positive class) is available
precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_proba)

# Plot Precision-Recall curve
plt.plot(recall, precision, color='teal')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

"""**-------------------------------------------------------- DONE --------------------------------------------------------**"""