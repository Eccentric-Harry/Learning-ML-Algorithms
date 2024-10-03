# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', gamma='scale', C=1.0)  # Radial basis function kernel
svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

logreg = LogisticRegression(max_iter=10000)  # Increased max_iter to ensure convergence
logreg.fit(X_train_scaled, y_train)

y_pred_logreg = logreg.predict(X_test_scaled)

logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {logreg_accuracy * 100:.2f}%")

if svm_accuracy > logreg_accuracy:
    print(f"SVM performed better with an accuracy of {svm_accuracy * 100:.2f}%")
else:
    print(f"Logistic Regression performed better with an accuracy of {logreg_accuracy * 100:.2f}%")
