import numpy as np

# Define the sample classes
# Samples for class ω1
X1 = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])
y1 = np.zeros(len(X1))  # class 0

# Samples for class ω2
X2 = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])
y2 = np.ones(len(X2))  # class 1

# Combine the samples
X = np.vstack((X1, X2))  # Combine feature values
y = np.hstack((y1, y2))  # Combine labels

# Compute the mean vectors of each class
mu1 = np.mean(X1, axis=0)  # Mean vector of class ω1
mu2 = np.mean(X2, axis=0)  # Mean vector of class ω2

print(f'Mean vector for class ω1: {mu1}')
print(f'Mean vector for class ω2: {mu2}')

# Compute the within-class scatter matrix (S)
S1 = np.dot((X1 - mu1).T, (X1 - mu1))  # Scatter matrix for class ω1
S2 = np.dot((X2 - mu2).T, (X2 - mu2))  # Scatter matrix for class ω2
S = S1 + S2  # Total within-class scatter matrix

print(f'Within-class scatter matrix S:\n{S}')

# Compute the difference of the mean vectors
mean_diff = mu1 - mu2

# Compute the weight vector w = S^(-1) * (mu1 - mu2)
S_inv = np.linalg.inv(S)
w = np.dot(S_inv, mean_diff)

print(f'The weight vector w is: {w}')
