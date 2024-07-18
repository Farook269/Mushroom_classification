import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
dir = "Mushroom_classification.xlsx"
data = pd.read_excel(dir)
data.drop_duplicates(inplace=True)
data.head()
data.info()
data.duplicated().sum()
data.dropna
y = data.iloc[:, -1]  # only first column
y = y.ravel()
X = data.iloc[:, 0:-2]  # All columns except first column
print(data["class"].unique())
data["class"] = [0 if i == "p" else 1 for i in data["class"]]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = pd.get_dummies(X)
print(data["class"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=2)


knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


scores = cross_val_score(knn, X, y, cv=5)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(scores) + 1),scores, marker='o', linestyle='-', color='r')
plt.title('Cross-Validation Scores of KNN')
plt.xlabel('Fold')
plt.ylabel('Accuracy')

plt.grid(False)
plt.show();
print(scores)

y_pred_knn = knn.predict(X_test)
y_true_knn = y_test
cm = confusion_matrix(y_true_knn, y_pred_knn)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_knn")
plt.ylabel("y_true_knn")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


rf = RandomForestClassifier(n_estimators=100, random_state=42)


rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {rf_accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

scores1 = cross_val_score(rf, X, y, cv=5)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='r')
plt.title('Cross-Validation Scores of Random Forest')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.grid(False)
plt.show();
print(scores1)

y_pred_rf = rf.predict(X_test)
y_true_rf = y_test
cm = confusion_matrix(y_true_rf, y_pred_rf)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_rf")
plt.ylabel("y_true_rf")
plt.show()




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)


y_pred = log_reg.predict(X_test)


lr_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {lr_accuracy * 100:.2f}%")


print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


scores2 = cross_val_score(log_reg, X, y, cv=5)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='r')
plt.title('Cross-Validation Scores of Logistic Regression')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.grid(False)
plt.show();
print(scores2)

y_pred_log_reg = log_reg.predict(X_test)
y_true_log_reg = y_test
cm = confusion_matrix(y_true_log_reg, y_pred_log_reg)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_log_reg")
plt.ylabel("y_true_log_reg")
plt.show()




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


svm = SVC()
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)


svm_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {svm_accuracy * 100:.2f}%")


print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


scores3 = cross_val_score(svm, X, y, cv=5)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='r')
plt.title('Cross-Validation Scores of SVM')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.grid(False)
plt.show();
print(scores3)

y_pred_svm = svm.predict(X_test)
y_true_svm = y_test
cm = confusion_matrix(y_true_svm, y_pred_svm)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_svm")
plt.ylabel("y_true_svm")
plt.show()

scores_knn = cross_val_score(knn, X, y, cv=5)
scores_rf = cross_val_score(rf, X, y, cv=5)
scores_log_reg = cross_val_score(log_reg, X, y, cv=5)
scores_svm = cross_val_score(svm, X, y, cv=5)

import numpy as np
mean_knn = np.mean(scores_knn)
mean_rf = np.mean(scores_rf)
mean_log_reg = np.mean(scores_log_reg)
mean_svm = np.mean(scores_svm)

print(f"KNN: Mean={mean_knn:.4f}")
print(f"Random Forest: Mean={mean_rf:.4f}")
print(f"Logistic Regression: Mean={mean_log_reg:.4f}")
print(f"SVM: Mean={mean_svm:.4f}")


mean_scores = {
    'KNN': mean_knn,
    'Random Forest': mean_rf,
    'Logistic Regression': mean_log_reg,
    'SVM': mean_svm
}

best_model = max(mean_scores, key=mean_scores.get)
print(f"Best model: {best_model} with a mean score of {mean_scores[best_model]:.4f}")
models = list(mean_scores.keys())
scores = list(mean_scores.values())

plt.figure(figsize=(10, 6))
plt.plot(models, scores, marker='o', linestyle='-', color='b')
plt.title('Mean Cross-Validation Scores of Different Models')
plt.xlabel('Models')
plt.ylabel('Mean Score')

plt.grid(False)
plt.show();
