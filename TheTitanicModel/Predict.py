# Introducing the notebook that was used for training the model
import ipynb.fs.full.Model

# Random Choice selector
from secrets import *

#
import pandas as pd

# Loading the model
from sklearn.externals import joblib

from TheTitanicModel.Predictions import train


# Made it so that in details your gender would remain the same
def get_sex():
    global gender
    global sex
    sex = 0
    gender = input('Enter your Gender: Male or Female\n').upper()
    if gender == 'MALE':
        sex = 0
    elif gender == 'FEMALE':
        sex = 1

# Made it so that in details your age would remain the same
def get_age():
    global age
    global realage
    age = 0
    realage = int(input('Enter Your Age\n'))
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


# Made it so that in details your family size would remain the same
def get_familysize():
    global family_size
    global realfamilysize
    family_size = 0.0
    realfamilysize = input('Your family size\n')
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


def columns():
        global name
        global pclass

        name = input('Kindly Input Your Name\n')

        # Passenger Class
        pclass = int(input('Enter Your Class: 1,2,or 3\n'))

        # Gender
        get_sex()

        # Age
        get_age()

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
        get_familysize()

        # Randomly pick a choice from Uniqe entries in Title column
        global title
        title = train['Title'].unique()
        title = choice(title)
        return


def predict():
        # Loading the trained model
        global prediction
        mymodel = joblib.load('TitanicSurvival.joblib')
        prediction = mymodel.predict([[pclass, sex, age, fare, cabin, embarked, family_size, title]])

        interactive()
        return

# Under Development: This function is expected to provide the opposite possible scenarios
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


columns()
predict()
# scenarios()
