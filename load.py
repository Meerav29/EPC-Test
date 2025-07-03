import re
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv('updated-questions-580.csv')
print(df.head())
print(df.shape)
print(df['Category'].value_counts())


def clean(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()

df['clean_q'] = df['Sample Question'].apply(clean)

X = df['clean_q']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(solver="liblinear", max_iter=1000)),
])

param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_df": [0.75, 0.9],
    "clf__C": [0.1, 1.0, 10.0],
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

y_pred = grid.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import numpy as np
errors = X_test[y_test != y_pred]
preds  = y_pred[y_test != y_pred]
trues  = y_test[y_test != y_pred]
for text, true, pred in zip(errors, trues, preds):
    print(f"True: {true}  ── Pred: {pred}\n  » {text}\n")
