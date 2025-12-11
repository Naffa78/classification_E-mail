import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\hp\Desktop\spam_assassin.csv')
data.columns = ['email', 'target']

X = data['email']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_vec, y_train)

y_pred = knn.predict(X_test_vec)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
error_rate = 1 - accuracy

print("Accuracy:", accuracy * 100)
print("Error Rate:", error_rate * 100)

counts = data['target'].value_counts()
counts.plot(kind='bar')
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Spam vs Non-Spam')
plt.show()
