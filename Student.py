'''
Author : Shivnikar Sudeep 
Admission No. : 15JE001152
Date : October 30 , 2017 
Problem : Predicting whether a student will pass or fail depending on his/her habits & previous record
Dataset Credits : https://archive.ics.uci.edu/ml/datasets/Student+Performance

'''
# importing required libraries 

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# setting variables for importing formatted dataset stored in Student.data file having 33 attributes whose names are listed in 'names' 
 
url = 'Student.data'
names = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences','G1','G2','G3']

#importing the dataset as dataframe using pandas and describing it

dataset = pandas.read_csv(url,names = names)
print('Dimensions of dataset are ',dataset.shape)
print ("There are total",dataset.shape[0],"instances and total",dataset.shape[1],"attributes in the dataset")
print("\nAttributes are as follows : ")
print(tuple(dataset.columns))

# If student scores less than 10 then he fails otherwise passes . This function is used to convert G3 into yes/no
def convert(g3):
    if(g3>=10):
        return 1
    else :
        return 0

dataset['G3'] = dataset['G3'].apply(convert)

# This function converts all the yes/no columns to 1/0
def yes_or_no(parameter):
    if parameter == 'yes' :
        return 1
    else :
        return 0

def yn(c) :
    dataset[c] = dataset[c].apply(yes_or_no)
    
#These columns have entries in yes/no format 
col = ['schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']

for c in col :
    yn(c)

# Let 0 denote student studies at Gabriel Pereira school and 1 denote that he tudies at Mousinho da Silveira school
school_to_int = {'GP':0 ,'MS':1}
dataset['school'] = dataset['school'].apply(lambda x : school_to_int[x] )

# Let 1 denote that student is Male and 0 denote that student is Female
sex_to_int = {'M':1,'F':0}
dataset['sex'] = dataset['sex'].apply(lambda x : sex_to_int[x])

# Let 1 denote that student lives in urban area and 0 denotes that student lives in rural area 
address_to_int = {'U' : 1 , 'R' : 0 }
dataset['address'] = dataset['address'].apply(lambda x : address_to_int[x])

# Let 1 denote that student's family size is greater than 3 and 1 otherwise
famsize_to_int = {'GT3' : 1 , 'LE3' : 0 }
dataset['famsize'] = dataset['famsize'].apply(lambda x : famsize_to_int[x])

# Let 1 denote that students parents live apart and 0 denote that they live together
Pstatus_to_int = {'A':1,'T':0}
dataset['Pstatus'] = dataset['Pstatus'].apply(lambda x : Pstatus_to_int[x])

# Let 0 denotes students parent is a teacher 
# Let 2 denotes students parent has 9-5 service 
# Let 3 denotes students parent is at home
# Let 4 denotes students parent is working in heath sector and 1 otherwise   
job = {'teacher':0 , 'other':1,'services':2,'at_home':3 , 'health':4}
dataset['Mjob'] = dataset['Mjob'].apply(lambda x : job[x] )
dataset['Fjob'] = dataset['Fjob'].apply(lambda x : job[x] )

# Let 0 denotes that student joined collage since it is near to his home
# Let 1 denote that student has joined college due to it's reputation
# Let 2 denote that student has joined college due to it's course structure 
# Let 3 denote some other reason of joining college
reason_to_int = {'home':0,'reputation':1,'course':2,'other':3}
dataset['reason'] = dataset['reason'].apply(lambda x : reason_to_int[x])

#Let 1 denote that father is guardian of student 
# Let 0 denote that mother is the guardian of student
# Let 2 denote the other cases
guardian_to_int = {'mother':0,'father':1,'other':2}
dataset['guardian'] = dataset['guardian'].apply(lambda x : guardian_to_int[x])

# Obtaining the co-relation matrix with pearsons measure for the dataset
corr = dataset.corr('pearson')    
all_columns = list(dataset.columns[:-1])
columns_to_drop = []

# Dropping the columns whose co-realtion coefficient with last column is less than 0.05
# This is done to improve the accuracy of prediction
print("\nColumns that are dropped are : ")
for i in all_columns :
    if abs(corr[i]['G3']) < 0.05 :
        columns_to_drop.append(i)
print(tuple(columns_to_drop))
for i in columns_to_drop :
    dataset.drop(i, axis = 1, inplace = True)

# Setting parameters for splitting the dataset into train and test
array = dataset.values
X = array[:,0:22]
Y = array[:,22]
validation_size = 0.20
seed = 7
scoring = 'accuracy'

# Splitting dataset into train and test data
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)

# Making list of all models 
models = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('SVM',SVC()))

print("\nAlgo :  Mean  (Std. Dev)")
results = []
names = []
bmodel_name = 'AAAA'
bmodel_mean = 0.00

# Finding the best model 
for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring = scoring)
    results.append(cv_results)
    names.append(name)
    mean = cv_results.mean()
    if(mean > bmodel_mean) :
        bmodel_name = name
        bmodel_mean = mean
        bmodel = model
    msg = "%s : %f (%f)" %(name,mean,cv_results.std())
    print (msg)

print("\n",bmodel_name,"is the best algorithm for computing the results")

# Applying the best model on the dataset
print("\nApplying",bmodel_name,"for prediction") 
sv = model
sv.fit(X_train, Y_train)
predictions = sv.predict(X_validation)
print("Accuracy using",bmodel_name,"on validation data : ",100*accuracy_score(Y_validation, predictions),"%")

# printing the confusion matrix 
print ("\n\nConfusion Matrix : ")
print(confusion_matrix(Y_validation, predictions))