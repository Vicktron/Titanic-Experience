from django.shortcuts import render
# from .apps import WebappConfig

from django.http import HttpResponse, JsonResponse
# My First Approach
# from django.shortcuts import get_object_or_404
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import statistics
# from .apps import WebappConfig

# from Prediction import Predict
# Introducing the notebook that was used for training the model
# from Prediction.Predictions import * I dont need this for the mean time
from sklearn.externals import joblib
import os
from io import StringIO

# Random Choice selector
from secrets import *

#
import pandas as pd

# Loading the model
from sklearn.externals import joblib

# from Prediction.Predictions import train

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from django_pandas.io import read_frame


train = pd.read_csv(('tittrain.csv'))
test = pd.read_csv(('tittest.csv'))

print(train.head())


train_test_combined = [train, test]

for dataset in train_test_combined:
    dataset["Title"] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 3, 'Rev': 3, 'Col': 3, 'Dona': 3,
                 'Major': 3, 'Mlle': 2, 'Mme': 1, 'Countess': 1, 'Sir': 3, 'Jonkheer': 1, 'Ms': 1, 'Don': 1, 'Capt': 1,
                 'Lady': 1
                 }

for dataset in train_test_combined:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# Since we derived Title from Name, we can remove the Name feature from the Datasets
train.drop(["Name"], axis=1, inplace=True)
test.drop(["Name"], axis=1, inplace=True)


sex_mapping = {'male': 0, 'female': 1}
for dataset in train_test_combined:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# there are Null values in Age column. impute Age values
# using median values for age based on Title to impute

train['Age'].fillna(train.groupby('Title')['Age'].transform("median"), inplace=True)
check = test['Age'].fillna(test.groupby('Title')['Age'].transform("median"), inplace=True)
print(check)

for dataset in train_test_combined:
    dataset.loc[(dataset["Age"] <= 16), 'Age'] = 0
    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 26), 'Age'] = 1
    dataset.loc[(dataset["Age"] > 26) & (dataset["Age"] <= 36), 'Age'] = 2
    dataset.loc[(dataset["Age"] > 36) & (dataset["Age"] <= 62), 'Age'] = 3
    dataset.loc[(dataset["Age"] > 62), 'Age'] = 4


# Fill out missing value for Embark as S as majority of data set is from S
for dataset in train_test_combined:
    dataset['Embarked'].fillna('S', inplace=True)


embark_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_combined:
    dataset["Embarked"] = dataset["Embarked"].map(embark_mapping)


# Fare
train["Fare"].fillna(train.groupby('Pclass')['Fare'].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby('Pclass')['Fare'].transform("median"), inplace=True)

# %%

for dataset in train_test_combined:
    dataset.loc[dataset["Fare"] <= 17, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 17) & (dataset["Fare"] <= 30), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 30) & (dataset["Fare"] <= 100), "Fare"] = 2
    dataset.loc[dataset["Fare"] > 100, "Fare"] = 3


for dataset in train_test_combined:
    dataset["Cabin"] = dataset["Cabin"].str[:1]

cabin_mapping = {'A': 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2.0, "G": 2.4, "T": 2.8}
for dataset in train_test_combined:
    dataset["Cabin"] = dataset["Cabin"].map(cabin_mapping)


train.Cabin.fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test.Cabin.fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# FamilySize
train["FamilySize"] = train.Parch + train.SibSp + 1
test["FamilySize"] = test.Parch + test.SibSp + 1

family_mapping = {1: 0.0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2.0, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_combined:
    dataset["FamilySize"] = dataset["FamilySize"].map(family_mapping)


drop_features = ['Parch', 'SibSp', 'Ticket']
for dataset in train_test_combined:
    dataset.drop(drop_features, axis=1, inplace=True)


# Train Data
train_data = train.drop(["PassengerId", "Survived"], axis=1)
target = train["Survived"]
test_data = test.drop(["PassengerId"], axis=1)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data, target, test_size=0.3, random_state=40)


from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt', criterion="entropy", max_depth=6,
                               oob_score=True)
model.fit(X_train, y_train)
joblib.dump(model, 'TitanicSurvival.pkl')


def index(request):
    context = {'a': 'Hello World'}
    return render(request, 'index.html', context)
    # return HttpResponse({'a': 1})

def get_sex(request):
    global gender
    global sex
    sex = 0
    gender = request.POST.get('sex').upper()
    if gender == 'MALE':
        sex = 0
    elif gender == 'FEMALE':
        sex = 1
    return render(request, 'index.html', sex)

# Made it so that in details your age would remain the same
def get_age(request):
    global age
    global realage
    age = 0
    realage = request.POST.get('age')
    if realage <= 16:
        age = 0
    elif 16 < age <= 26:
        age = 1
    elif 26 < age <= 36:
        age = 2
    elif 36 < age <= 62:
        age = 3
    elif age > 62:
        age = 4
    return render(request, 'index.html', age)


# Made it so that in details your family size would remain the same
def get_familysize(request):
    global family_size
    global realfamilysize
    family_size = 0.0
    realfamilysize = request.POST.get('familysize')
    if realfamilysize == 1:
        family_size = 0.0
    elif realfamilysize == 2:
        family_size = 0.4
    elif realfamilysize == 3:
        family_size = 0.8
    elif realfamilysize == 4:
        family_size = 1.2
    elif realfamilysize == 5:
        family_size = 1.6
    elif realfamilysize == 6:
        family_size = 2.0
    elif realfamilysize == 7:
        family_size = 2.4
    elif realfamilysize == 8:
        family_size = 2.8
    elif realfamilysize == 9:
        family_size = 3.2
    elif realfamilysize == 10:
        family_size = 3.6
    elif realfamilysize == 11:
        family_size = 4.0
    return render(request, 'index.html', familysize)


def interactive():
    global status
    # Making the output interactive
    if prediction == [1]:
        status = '"SURVIVED"'
    else:
        status = '"DID NOT SURVIVE'

    details = pd.DataFrame({'Passenger Class': [pclass], 'Gender': [gender], 'Age': [realage], 'Fare': [fare],
                            'Cabin': [cabin], 'Embarked': [embarked], 'Family Size': [realfamilysize],
                            'Title': [title]})

    print(f'Well {name} it turns out that you {status} the Titanic Sinking\n')
    print(f'Your Details Were as Follows\n{details}')
    return


def columns(request):
        global name
        global pclass

        name = request.POST.get('name')

        # Passenger Class
        pclass = request.POST.get('pclass')

        # Gender
        get_sex(request)

        # Age
        get_age(request)

        # Randomly pick a choice from Unique entries in Fare column
        global fare
        fare = train['Fare'].unique()
        fare = choice(fare)

        # Randomly pick a choice from Unique entries in Cabin column
        global cabin
        cabin = train['Cabin'].unique()
        cabin = choice(cabin)

        # Randomly pick a choice from Unique entries in Embarked column
        global embarked
        embarked = train['Embarked'].unique()
        embarked = choice(embarked)

        # FamilySize
        get_familysize(request)

        # Randomly pick a choice from Uniqe entries in Title column
        global title
        title = train['Title'].unique()
        title = choice(title)
        return render(request, 'index.html', name, pclass)



def predict():
        # Loading the entries into  the  trained model
        global prediction
        mymodel = joblib.load('TitanicSurvival.joblib')
        prediction = mymodel.predict([[pclass, sex, age, fare, cabin, embarked, family_size, title]])

        # interactive()
        return

# Under Development: This function is expected to provide the opposite possible scenarios
# New IDEA: Provide suggestions to survive by providing 3 possible scenerios
# def scenarios():
#     if status == '"SURVIVED"':
#         for cases in train['PassengerId'], train['Pclass'], train['Sex'], train['Age'],\
#                      train['Fare'], train['Cabin'], train['Embarked']:
#             print(train['Survived'] != 1)
#     elif status == '"DID NOT SURVIVED"':
#         for cases in train['PassengerId'], train['Pclass'], train['Sex'], train['Age'],\
#                      train['Fare'], train['Cabin'], train['Embarked']:
#             print(train['Survived'] != 0)
#     return


def PredictFate(request):
    if request.method == 'POST':
        columns()
        predict()
        fate = interactive()
        context = {'fate': fate}

    return render(request, 'index.html', context)

# PredictFate(request)
# My First Approach
# class call_model(APIView):
#
#     def get(self, request):
#         if request.method == 'GET':
#
#             # Gets the user input
#             params = request.GET.get(columns())
#
#             # Predict the fate of the user
#             response = WebappConfig.predictor.predict(params)
#
#             # Return a JSON response
#             return JsonResponse(response)

