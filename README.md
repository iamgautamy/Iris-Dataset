# Iris-Dataset

## About the Iris Dataset

Originally published at UCI Machine Learning Repository: Iris Data Set, this small dataset from 1936 is often used for testing out machine learning algorithms and visualizations (for example, Scatter Plot). Each row of the table represents an iris flower, including its species and dimensions of its botanical parts, sepal and petal, in centimeters

Iris flower is divided into 3 species:

    Iris setosa
    Iris versicolor
    Iris virginica

The iris dataset consists of 4 features:

    Sepal Length
    Sepal Width
    Petal Length
    Petal Width

The objective of this project is to predict the species given the four features of an iris flower.

## Load Dataset

    name = ['sepal-length','sepal-width','petal-length','petal-width','class']
    dataset = read_csv('Iris.csv', header=0, names=name)
    
## Visualizing the Dataset

### Histogram of each attributes


![Histogram_attributes](https://user-images.githubusercontent.com/46325271/141672065-3a2b8ff7-6d59-4b80-baea-d71e903d1ad1.png)

### Scatter Matrix diagram between the attributes

![Scatter_Matrix](https://user-images.githubusercontent.com/46325271/141672102-fa0a3699-0852-4064-aac3-89b1a1d8c3c6.png)

### Swarmplot between each attributes

#### Sepal Length

![Swarmplot_SL](https://user-images.githubusercontent.com/46325271/141672156-baa3cd47-7a9b-48b8-bc0f-ea9d94ef11c5.png)

#### Sepal Width

![Swarmplot_SW](https://user-images.githubusercontent.com/46325271/141672178-aa20b11e-c23c-463e-b28b-b7e44f9a7aaa.png)

#### Petal Length

![Swarmplot_PL](https://user-images.githubusercontent.com/46325271/141672194-9045caf9-783a-475c-9b92-96905f276a5a.png)

#### Petal Width

![Swarmplot_PW](https://user-images.githubusercontent.com/46325271/141672208-4e517731-4586-48ff-854e-c208772acbc8.png)


## Evaluating Models

### Splitting Data into Train and Test sets

    array = dataset.values
    X = array[:,0:4]
    y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.3, random_state=None)
    
Above code splits data between last column and remaining columns as X and y. It is then further splitted into training and testing dataset using train_test_split()

### Testing Different Algorithms

#### Support Vector Machines (SVM)

    model_SVM = SVC()
    model_SVM.fit(X_train, Y_train)
    prediction = model_SVM.predict(X_validation)
    print('The accuracy of the SVM is: ', accuracy_score(prediction, Y_validation))
The accuracy of the SVM is:  0.9555555555555556

#### Logistic Regression 

    model_LR = LogisticRegression()
    model_LR.fit(X_train, Y_train)
    prediction = model_LR.predict(X_validation)
    print('The accuracy of Logistic Regression is: ', accuracy_score(prediction, Y_validation))
The accuracy of Logistic Regression is:  0.9333333333333333

#### Decision Tree 

    model_DT = DecisionTreeClassifier()
    model_DT.fit(X_train, Y_train)
    prediction = model_DT.predict(X_validation)
    print('The accuracy of Decision Tree is: ',accuracy_score(prediction, Y_validation))
The accuracy of Decision Tree is:  0.9333333333333333

#### K-Nearest Neighbors

    model_KN = KNeighborsClassifier(n_neighbors=6)
    model_KN.fit(X_train, Y_train)
    prediction = model_KN.predict(X_validation)
    print('The accuracy of KNN is: ', accuracy_score(prediction, Y_validation))
The accuracy of KNN is:  0.9555555555555556

## Making Predictions

### Selecting SVM model as per above evaluation

    model = SVC()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_validation)
    print('The accuracy of the SVM is: ', accuracy_score(prediction, Y_validation))
The accuracy of the SVM is:  0.9555555555555556

## Evaluating Predictions

    print(accuracy_score(Y_validation, prediction))
    print(confusion_matrix(Y_validation, prediction))
    print(classification_report(Y_validation, prediction))

               0.9555555555555556
            [[17  0  0]
             [ 0 15  1]
             [ 0  1 11]]
                             precision    recall  f1-score   support

                Iris-setosa       1.00      1.00      1.00        17
            Iris-versicolor       0.94      0.94      0.94        16
             Iris-virginica       0.92      0.92      0.92        12

                   accuracy                           0.96        45
                  macro avg       0.95      0.95      0.95        45
               weighted avg       0.96      0.96      0.96        45




