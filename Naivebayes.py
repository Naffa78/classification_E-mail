import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

file_path = r'C:\Users\ac\Downloads\spam_assassin.csv.zip'
data = pd.read_csv(file_path)

X = data['text']
y = data['label']

enc = LabelEncoder()
y_enc = enc.fit_transform(y)

vec = TfidfVectorizer()
X_vec = vec.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_enc, test_size=0.2, random_state=42
)

nb = GaussianNB()
nb.fit(X_train.toarray(), y_train)
pred = nb.predict(X_test.toarray())

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()

acc = (tp + tn) / (tp + tn + fp + fn)
err = 1 - acc

print("Accuracy:", acc * 100)
print("Error Rate:", err * 100)

cnt = data['target'].value_counts()

plt.figure(figsize=(8, 5))
cnt.plot(kind='bar')
plt.title('Email Classes')
plt.xlabel('Target')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
