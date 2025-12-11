import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n = len(self.classes)
        m = X.shape[1]

        self.mean = np.zeros((n, m))
        self.var = np.zeros((n, m))
        self.prior = np.zeros(n)

        for i, c in enumerate(self.classes):
            x_c = X[y == c]
            self.mean[i] = x_c.mean(axis=0)
            self.var[i] = x_c.var(axis=0)
            self.prior[i] = x_c.shape[0] / X.shape[0]

    def predict(self, X):
        pred = []
        for row in X:
            pred.append(self._predict_row(row))
        return np.array(pred)

    def _predict_row(self, x):
        vals = []
        for i, c in enumerate(self.classes):
            p = np.log(self.prior[i])
            cond = np.sum(np.log(self._pdf(i, x)))
            vals.append(p + cond)
        return self.classes[np.argmax(vals)]

    def _pdf(self, idx, x):
        mean = self.mean[idx]
        var = self.var[idx]
        num = np.exp(-((x - mean)**2) / (2 * var + 1e-9))
        den = np.sqrt(2 * np.pi * var + 1e-9)
        return num / den


data = pd.read_csv(r'C:\Users\hp\Desktop\spam_assassin.csv')

d = data.sample(frac=0.5, random_state=42)

X = d['text']
y = d['target']

vec = TfidfVectorizer(max_features=5000)
Xv = vec.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    Xv, y, test_size=0.2, random_state=42, stratify=y
)

nb = NaiveBayes()
nb.fit(X_train.toarray(), y_train)
pred = nb.predict(X_test.toarray())

cm = confusion_matrix(y_test, pred)
print(cm)
print(classification_report(y_test, pred))

tn, fp, fn, tp = cm.ravel()
acc = (tp + tn) / (tp + tn + fp + fn)
err = 1 - acc

print("Accuracy:", acc*100)
print("Error Rate:", err*100)

cnt = data['target'].value_counts()
cnt.plot(kind='bar')
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Spam Data')
plt.show()
