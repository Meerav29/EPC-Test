import re
import pandas as pd
from sklearn.model_selection import train_test_split
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
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_df=0.9)),
    ("clf", LogisticRegression(solver="liblinear", C=1.0)),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))