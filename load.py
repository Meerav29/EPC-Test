import pandas as pd
import re 


df = pd.read_csv('questions.csv')
print(df.head())
print(df.shape)
print(df['Category'].value_counts())


def clean(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()

df['clean_q'] = df['Sample Question'].apply(clean)
