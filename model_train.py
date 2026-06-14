import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv(r"C:\Users\admin\loan_approval_large.csv")
features = ['Age','Income','LoanAmount','EmploymentYears','ExistingDebt']
x = dataset[features].values
y = dataset['Approved'].values

from sklearn.model_selection import train_test_split
x_train, x_test,  y_train ,  y_test = train_test_split(x , y , test_size= 0.25, random_state =0)

print(x_train)
print(x_test)
print(y_train)
print(y_test)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(x_train.shape)
print(x_test.shape)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state =0)
dt.fit(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10,  random_state=0)
rf.fit(x_train, y_train)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0,  max_iter=1000)
lr.fit(x_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric =  'minkowski', p=2)
knn.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)
y_pred_rf = rf.predict(x_test)
y_pred_lr = lr.predict(x_test)
y_pred_knn = knn.predict(x_test)

from sklearn.metrics import accuracy_score

print("\nModel Accuracies:")
print("-------------------")
print("KNN:", accuracy_score(y_test, y_pred_knn))
print("Logistic:", accuracy_score(y_test, y_pred_lr))
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))

scores = {
    "KNN": accuracy_score(y_test, y_pred_knn),
    "Logistic": accuracy_score(y_test, y_pred_lr),
    "Decision Tree": accuracy_score(y_test, y_pred_dt),
    "Random Forest": accuracy_score(y_test, y_pred_rf)
}

best_model = max(scores, key=scores.get)

print("Best Model:", best_model)
print("Best Accuracy:", scores[best_model])


import pickle

if best_model == "KNN":
    pickle.dump(knn, open("model.pkl","wb"))

elif best_model == "Logistic":
    pickle.dump(lr, open("model.pkl","wb"))

elif best_model == "Decision Tree":
    pickle.dump(dt, open("model.pkl","wb"))

else:
    pickle.dump(rf, open("model.pkl","wb"))


pickle.dump(sc, open("scaler.pkl","wb"))