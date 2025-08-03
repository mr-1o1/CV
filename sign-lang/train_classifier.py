import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

def printp(*args):
    print("="*80)
    print(*args)
    print()

data_dict = pickle.load(open('./data.pickle', 'rb'))

print(data_dict.keys())
# print(data_dict)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

printp("Splitting the data")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

printp("Training the data")
model.fit(x_train, y_train)

printp("Creating predictions")
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
printp(f"{score*100:0.2F}% of samples were classified correctly!")

