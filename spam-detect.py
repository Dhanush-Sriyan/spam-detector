import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('spam.csv', encoding='latin-1') #stor file data in data variable
data= data[['v1','v2']]  #Only select 2 columns and others remove
data.columns=['label', 'text']  #columns rename
data['label']=data['label'].map({'ham':0, 'spam':1})   #label column convert to numaric 

# Download NLTK data (only needed once)
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))  # create stop word set "the, is , in ,....."

def clean_text(text): #data clean function
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub('[^a-zA-Z]', ' ', text) #'[^a-zA-Z]'= mean all caracters not leters
    # Tokenize and remove stopwords + stem
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

data['clean_text'] = data['text'].apply(clean_text)

# Convert text to numbers
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(data['clean_text']).toarray()
y = data['label']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained successfully! ✅")


# Predictions
y_pred = model.predict(X_test)

# Scores
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(data['label'].value_counts())
print(data.head)

import pickle

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("Model Saved Successfully!")