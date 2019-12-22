import os
from flask import Flask, abort, session, request, redirect
from flask.json import jsonify
import sqlite3
#import ibm_db_dbi

#connection = ibm_db_dbi.connect("dbname = 'BLUDB', user = 'jgt30606', host = 'dashdb-txn-sbox-yp-lon02-01.services.eu-gb.bluemix.net', password = 'ds-gmd1c01dkb68f', port = '50000'")
connection = sqlite3.connect('server/data.db')
#connection.autocommit = True
cursor = connection.cursor()
create_table_command = "CREATE TABLE IF NOT EXISTS USER(NAME VARCHAR2(40), EMAIL VARCHAR2(30), PASSWORD VARCHAR(15));"
cursor.execute(create_table_command)
connection.commit()


app = Flask(__name__,  static_url_path='')

from server.routes import *
from server.services import *

initServices(app)

import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

male_entries = ['M', 'm', 'Male', 'male', 'Cis Male', 'Cis Man', 'Guy (-ish) ^_^', 'Mail', 'Make', 'Mal', 'Male ', 'Male (CIS)', 'Male-ish', 'Malr', 'Man', 'cis male', 'maile', 'msle']
female_entries = ['Cis Female', 'F', 'Femake', 'Female', 'Female ','Female (cis)','Female (trans)', 'Trans woman', 'Trans-female', 'Woman', 'cis-female/femme', 'f', 'femail', 'female', 'woman' ]
other_entries = ['Agender', 'A little about you', 'All', 'Androgyne', 'Enby', 'Genderqueer','Nah', 'Neuter', 'fluid', 'male leaning androgynous', 'non-binary', 'ostensibly male, unsure what that really means', 'p', 'queer', 'queer/she/they', 'something kinda male?']
def encodeGender(values):
    for index in range(values.size):
        if values[index] in male_entries:
            values[index] = 0
        elif values[index] in female_entries:
            values[index] = 1
        elif values[index] in other_entries:
            values[index] = 2

# Importing the dataset
dataset = pd.read_csv('server/OriginalDataset.csv')

#Only considering employed people
employees = dataset[dataset['self_employed'] == "No"]

#Encoding number of employees numerically
di = {"1-5": 3, "6-25": 16, "26-100":63, "100-500": 300, "500-1000": 750, "More than 1000": 2500 }
employees["no_employees"].replace(di, inplace=True)

def getDependentVariables():
    dependentVariables = employees.iloc[:, 7].values
    return dependentVariables

def getIndependentVariables():
    #Taking out irrelevant variables
    independentVariables = employees.iloc[:, [1,2,6,9,10,11,12,14,16,17,18]].values

    #Encode Gender
    encodeGender(independentVariables[:, 1])

    #Encoding Anonymity, leave, and mental Health Consequence
    labelencoder_independent = LabelEncoder()

    items = [2, 4, 5, 6, 7, 8, 9, 10]
    for i in items:
        independentVariables[:, i] = labelencoder_independent.fit_transform(independentVariables[:, i])

    #0,1,2 - Male| Female | Other 
    #3,4 - Family history (No) | Family history (Yes)
    #5,6 - Remote work (No) | Remote work (Yes)
    #7,8 - Tech company (No) | Tech company (Yes)
    #9,10,11 - Benefits (Dont know) | Benefits (No) | Benefits (Yes)
    #12,13,14 - Wellness Program (Dont know) | Wellness Program (No) | Wellness program (yes)
    #15,16,17 - Anonymity (Dont know ) | Anonymity (No) | Anonymity (yes)
    #18,19,20,21,22 - Leave  (Dont know) | Leave (Somewhat difficult) |Leave (Somewhat easy)|Leave (Very difficult) | Leave (Very Easy)
    #23,24,25 - Mental Health Conseuquence (Maybe) | MHC (No) |MHC (Yes)
    #26 - Age
    #27 - Number of employees
    onehotencoder = OneHotEncoder(categorical_features = [1,2,4,5,6,7,8,9,10])
    independentVariables = onehotencoder.fit_transform(independentVariables).toarray()
    return independentVariables


feature_to_index = {
    'age': 26,
    'noOfEmployees': 27
}
feature_to_value_to_index = {
    'gender': {
        'male': 0,
        'female': 1,
        'other': 2
    },
    'familyTrend': {
        'no': 3,
        'yes': 4
    },
    'remoteWork': {
        'no': 5,
        'yes': 6
    },
    'techCompany': {
        'no': 7,
        'yes': 8
    },
    'benefits': {
        'dontKnow': 9,
        'no': 10,
        'yes': 11
    },
    'wellnessProgram': {
        'dontKnow': 12,
        'no': 13,
        'yes': 14
    },
    'anonymity': {
        'dontKnow': 15,
        'no': 16,
        'yes': 17
    },
    'leave': {
        'dontKnow': 18,
        'somewhatDifficult': 19,
        'somewhatEasy': 20,
        'veryDifficult': 21,
        'veryEasy': 22
    },
    'mhc': {
        'maybe': 23,
        'no': 24,
        'yes': 25
    }
}

def getSignificantFeatureNames(significantFeatureIndices):
    l1 = [name for (name, index) in feature_to_index.items() if index in significantFeatureIndices]
    l2 = [name for (name, valueToIndexMap) in feature_to_value_to_index.items()
          if set(valueToIndexMap.values()).intersection(significantFeatureIndices)]
    return l1 + l2

def getIndependentVariablesFrom(person):
    b = [0] * 28
    for key, index in feature_to_index.items():
        if key in person:
            b[index] = person[key]

    for key, valueEnums in feature_to_value_to_index.items():
        if key not in person:
            continue
        actualValue = person[key]
        index = valueEnums[actualValue]
        b[index] = 1
    return b





independentVariables = getIndependentVariables()
dependentVariables = getDependentVariables()
#print (independentVariables)



# Train Test Split
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(independentVariables, dependentVariables, test_size=0.1,
                                                    random_state=0)

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 5)
rfe = rfe.fit(X_train, y_train)
# summarize the selection of the attributes
#print(rfe.support_)
#print(rfe.ranking_)

#print(rfe.ranking_)
rfe.ranking_ = [1.0 / float(x) for x in rfe.ranking_]

# plt.title("Feature importances")
#print(rfe.ranking_)
labels = ['Male', 'Female', 'Other', 'Family history (No)', 'Family history (Yes)',
          'Remote work (No)', 'Remote work (Yes)',
          'Tech company (No)', 'Tech company (Yes)',
          'Benefits (Dont know)', 'Benefits (No)', 'Benefits (Yes)',
          'Wellness Program (Dont know)', 'Wellness Program (No)', 'Wellness program (yes)',
          'Anonymity (Dont know )', 'Anonymity (No)', 'Anonymity (yes)',
          'Leave  (Dont know)', 'Leave (Somewhat difficult)', 'Leave (Somewhat easy)', 'Leave (Very difficult)',
          'Leave (Very Easy)',
          'Mental Health Conseuquence (Maybe)', 'MHC (No)', 'MHC (Yes)',
          'Age',
          'Number of employees']
# plt.bar(range(len(rfe.ranking_)), rfe.ranking_,
#         color="blue", align="center")
# plt.xticks(range(len(rfe.ranking_)), labels, rotation='vertical')
# # plt.xlim([-1, X.shape[1]])
# # plt.savefig(rankingPngFileLocation)
# plt.show()

# Selecting Significant Features
SIGNIFICANT_FEATURES = [0, 4, 11, 19, 21]
X_train = X_train[:, SIGNIFICANT_FEATURES]
X_test = X_test[:, SIGNIFICANT_FEATURES]

# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=400, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix, accuracy_score

#cm = confusion_matrix(y_test, y_pred)

#print("Accuracy Score:- ", accuracy_score(y_test, y_pred) * 100)

def predict(person):
    b = getIndependentVariablesFrom(person)
    choppedFeatures = [[b[i] for i in SIGNIFICANT_FEATURES]]
    prediction = classifier.predict(choppedFeatures)
    return prediction[0]


significant_features_of_current_model = getSignificantFeatureNames(SIGNIFICANT_FEATURES)

@app.route('/predict', methods=['POST'])
def get_prediction():
    #validate_json(required_keys=significant_features_of_current_model)

    person = request.json

    p = predict(person)
    if p == 'Yes':
        ans = "YOU REQUIRE MENTAL HEALTH COUNSELING"
    else:
        ans = "YOU DON'T REQUIRE MENTAL HEALTH COUNSELING"
    # TODO : actual code goes here
    return jsonify({'result': ans})

@app.route('/signup', methods=['POST'])
def signup():
    connection = sqlite3.connect('server/data.db')
    cursor = connection.cursor()
    person = request.json
    query="SELECT * FROM USER WHERE EMAIL=?"
    result=cursor.execute(query, (person["email"],))
    connection.commit()
    row=result.fetchone()
    if row:
        return jsonify({'result': "F"})
    else:
        query="INSERT INTO USER(NAME,EMAIL,PASSWORD) VALUES('" + person["name"] + "','" + person["email"] + "','" + person["password"] + "');"
        result=cursor.execute(query)
        connection.commit()
        connection.close()
        return jsonify({'result': "P"})

@app.route('/login', methods=['POST'])
def login():
    connection = sqlite3.connect('server/data.db')
    cursor = connection.cursor()
    person = request.json
    query="SELECT * FROM USER WHERE EMAIL=? AND PASSWORD=?"
    result=cursor.execute(query, (person["email"],person["password"],))
    row=result.fetchone()
    if row:
        return jsonify({'result': "P", 'name' : row[0]})
    else:
        return jsonify({'result': "F", 'name' : 'NULL'})

@app.route('/1', methods=['GET'])
def home():
    return "App Working"

'''
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT'))
'''

if 'FLASK_LIVE_RELOAD' in os.environ and os.environ['FLASK_LIVE_RELOAD'] == 'true':
    import livereload
    app.debug = True
    server = livereload.Server(app.wsgi_app)
    server.serve(port=os.environ['port'], host=os.environ['host'])
