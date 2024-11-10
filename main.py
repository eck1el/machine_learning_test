import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

#Este es un modelo clasico de clasificacion Logistic Regression
from sklearn.linear_model import LogisticRegression

#Este es el modelo de entrenamiento Random Forest
from sklearn.ensemble import RandomForestClassifier

#Este es el modelo de entrenamiento DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

#Este es el modelo de entrenamiento Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

#Este es el modelo de entrenamiento K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score

#Pasos de limpieza y preparación de datos
#1- Cargar el dataset
#2- Revisar valores nulos
    #Eliminar valores nulos o imputarse(llenarse con la media, mediana o moda)
#3-convertir variables categoricas a numeros(por ejemplo en la columna sex, Male = 0, Female = 1)
    # tambien se puede transformar utilizando one-hot encoding para convertir a binario
#4-Normalizacion o Escalado de los datos
#5- Eliminar columnas irrelevantes como Name, Ticket etc

def limpieza_datos():
    # Paso 1: Cargar el dataset
    titanic_cleaned = pd.read_csv(r"train.xls", sep=',')

    #paso 2: Revisar y manejar valores nulos
    #imputamos la columna Age con la mediana
    titanic_cleaned.fillna({'Age': titanic_cleaned['Age'].median()}, inplace=True)

    # Calcular y mostrar la mediana de la columna Age
    age_median = titanic_cleaned['Age'].median()
    #print(f"Mediana de la columna Age: {age_median}")

    #Imputamos la columna Embarked con la moda
    titanic_cleaned.fillna({'Embarked': titanic_cleaned['Embarked'].mode()[0]}, inplace=True)
    # Calcular y mostrar la moda de la columna Embarked
    embarked_mode = titanic_cleaned['Embarked'].mode()[0]
    #print(f"Moda de la columna Embarked: {embarked_mode}")

    #paso 3: convertir variables categoricas a numericas
    #convertir 'sex' a valores numericos
    titanic_cleaned['Sex'] = titanic_cleaned['Sex'].map({'male':0, 'female':1})

    #Convertir 'Embarked' a variables numericas
    titanic_cleaned['Embarked'] = titanic_cleaned['Embarked'].map({'S':1, 'C':2, 'Q':3})

    # Paso 4: Normalización/Escalado
    # Escalar las variables numéricas como 'Age' y 'Fare'
    #scaler = StandardScaler()
    #titanic_cleaned[['Age', 'Fare']] = scaler.fit_transform(titanic_cleaned[['Age', 'Fare']])

    # Paso 5: Eliminar columnas irrelevantes
    titanic_cleaned.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    return titanic_cleaned


def separarInformacion(titanic_cleaned):

    #Las caracteristicas las colocamos en el eje X
    x = titanic_cleaned[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

    #El target(lo que queremos predecir) siempre colocarlo en el eje Y
    y = titanic_cleaned["Survived"]
    print(f"Dataset titanic cargado. Numero de registros: {x.shape[0]}")
    print("-----------------------------------")
    dividimosPorcentajes_training_test(x, y)

def dividimosPorcentajes_training_test(x, y):
    #Dividimos en un 20% para test y 80%para training
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
    print(f"Hay {X_train.shape[0]} registros en el set de training y {X_test.shape[0]} registros en el set de test")
    print("-----------------------------------")
    definimosModeloYEntrenamosModelo(X_train, X_test, y_train, y_test)

def definimosModeloYEntrenamosModelo(X_train, X_test, y_train, y_test): #probamos el modelo Logistic Rregression
    #definimos el modelo, en este caso un modelo muy basico de regresion logistica
    model = LogisticRegression(max_iter=1000)

    #Entrenamos el modelo
    model.fit(X_train, y_train)

    train_score = round(model.score(X_train, y_train), 4)
    test_score = round(model.score(X_test, y_test), 4)

    #Si el accuracy se encuentra entre el 0.65 y 0.85 significa que el modelo entrenado esta generalizando bien
    #Si el accuracy se encuentra por encima de 0.85 o debajo de 0.65 significa que no ha entrenado bien y no sirve(indicador de overfitting o sobreajuste)
    print(f"Train Accuracy: {train_score}")
    print(f"Test Accuracy: {test_score}")
    print("-----------------------------------")
    entrenamiento_cross_validation_Logistic_regression(X_train, y_train)
    entrenamiento_cross_validation_Random_forest(X_train, y_train)
    entrenamiento_cross_validation_Decision_Tree(X_train, y_train)
    entrenamiento_cross_validation_Gaussian_Naive_Bayes(X_train, y_train)

#este es un modelo de entrenamiento que genera test dentro del porcentaje de datos definido para el entrenamiento

def entrenamiento_cross_validation_Logistic_regression(X_train, y_train): #probamos el modelo  de entrenamiento Logistic Rregression
    model = LogisticRegression(max_iter=1000)
    random_seed = 42 #Semilla para los aleatorios
    splits = 5 #Numero de cortes, segmentos o splits para KFolds
    kf = KFold(n_splits=splits, random_state=random_seed, shuffle=True)

    #Proceso de cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc')
    scores_mean_lr = scores.mean()


    #me va a mostrar una lista con los 5 valores de entrenamiento(splits) y la puntuacion promedio de cada uno
    print(f"Metricas accurancy cross_validation-Logistic Regression : {scores}")
    print(f"Promedio cross_validation-Logistic Regression : {scores_mean_lr}")

    #Entrenamos el modelo
    model.fit(X_train, y_train)

    #Comprobamos las metricas tras el entrenamiento
    print(f"Train Accuracy Logistic Regression: {model.score(X_train, y_train)}")
    print("-----------------------------------")

def entrenamiento_cross_validation_Random_forest(X_train, y_train): #probamos el modelo de entrenamiento random forest
    model = RandomForestClassifier()
    kf =  KFold(n_splits=5, random_state=42, shuffle=True)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc')
    scores_mean_rf = scores.mean()
    print(f"Promedio AUC score - Random Forest: {scores_mean_rf}")

    #Entrenamos el modelo
    model.fit(X_train, y_train)

    #Comprobamos las metricas tras el entrenamiento
    print(f"Train Accuracy Random Forest: {model.score(X_train, y_train)}")
    print("-----------------------------------")

def entrenamiento_cross_validation_Decision_Tree(X_train, y_train): #probamos el modelo de entrenamiento decision tree
    model = DecisionTreeClassifier()
    kf =  KFold(n_splits=5, random_state=42, shuffle=True)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc')
    scores_mean_rf = scores.mean()
    print(f"Promedio AUC score - Decision Tree: {scores_mean_rf}")

    #Entrenamos el modelo
    model.fit(X_train, y_train)

    #Comprobamos las metricas tras el entrenamiento
    print(f"Train Accuracy Decision Tree: {model.score(X_train, y_train)}")
    print("-----------------------------------")

def entrenamiento_cross_validation_Gaussian_Naive_Bayes(X_train, y_train): #probamos el modelo de entrenamiento Gaussian Naive Bayes
    model = GaussianNB()
    kf =  KFold(n_splits=5, random_state=42, shuffle=True)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc')
    scores_mean_rf = scores.mean()
    print(f"Promedio AUC score - Gaussian Naive Bayes: {scores_mean_rf}")

    #Entrenamos el modelo
    model.fit(X_train, y_train)

    #Comprobamos las metricas tras el entrenamiento
    print(f"Train Accuracy Gaussian Naive Bayes: {model.score(X_train, y_train)}")
    print("-----------------------------------")

def machine_learning():
    data = limpieza_datos()
    separarInformacion(data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    machine_learning()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
