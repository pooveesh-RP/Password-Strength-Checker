import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import getpass

data = pd.read_csv("data.csv", error_bad_lines=False)
print(data.head())

data = data.dropna()
data["strength"] = data["strength"].map({0: "Weak", 1: "Medium", 2: "Strong"})
print(data.sample(5))

def word(password):
    return [char for char in password]

x = np.array(data["password"])
y = np.array(data["strength"])

tdif = TfidfVectorizer(tokenizer=word)
x = tdif.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.05, random_state=42)

model = RandomForestClassifier()
model.fit(xtrain, ytrain)

print(model.score(xtest, ytest))

user = getpass.getpass("Enter Password: ")
data = tdif.transform([user]).toarray()
output = model.predict(data)

print(output)
