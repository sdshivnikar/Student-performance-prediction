# Student-performance-prediction
   This is my first project on Machine Learning.This was done just to get introduced to Machine Learing using Python using various powerful libraries  like Scipy , Numpy , Pandas etc. 
   
   However the accuracy was preety good nearly 93 % using Logistic Regression.
   
   As title suggests we predict whether a student will pass or fail in the upcoming examination using the details of the students obtained after a survey.
   
   Dataset credits goes to http://archive.ics.uci.edu/ml/datasets/Student+Performance. Dataset contains total 33 fields. Last field is G3 containing the final marks of the students. If a student scores 10 or more in the final exam then he considered as pass otherwise fail.
   
   Steps : 
   1. First Convert all the non-numerical values in the dataset into numerical values. For example assign 1 to yes and 0 to no for columns having yes/no entries.
   2. Then use Pearson's corelation co-efficient to eliminate the columns whose corelation coefficient with G3 is less than 0.05. Eliminating such columns increases the performance of the algorithm.
   3. Split the data into train and test data.
   4. Use the different models like Logistic Regression (LR) , Support Vector Machine (SVM) , Naive Bayes (NB) etc and apply them on the data. Use 10 fold cross validation and take its mean as accuracy measure.
   5. Since Logistic Regression has highest mean accuracy, fit the training data into model of Logistic Regression. Then predict the results for test data and compare the result with acual output. Print the confusion matrix and accuracy.
