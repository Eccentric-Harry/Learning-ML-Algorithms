# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = datasets.load_digits()

# Features and labels
X = digits.data
y = digits.target

# Split into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features for better performance of SVM and Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Support Vector Machine (SVM) model
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)

# Train the Logistic Regression model
logreg_model = LogisticRegression(max_iter=10000)
logreg_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_logreg = logreg_model.predict(X_test_scaled)

# Calculate the accuracy for both models
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

# Output the accuracies
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")
print(f"Logistic Regression Accuracy: {accuracy_logreg * 100:.2f}%")