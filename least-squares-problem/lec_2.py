import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the file
df = pd.read_csv("real_estate_dataset.csv")

# get the number of samples
n_samples, n_features = df.shape

# Get the coulmns
columns = df.columns

#Save the features in a text file
np.savetxt("coulmn_names.txt", columns, fmt="%s")


# features used => ["square feet", "garage", "location_score", "distance_to_center"]


X = df[["Square_Feet", "Garage_Size", "Location_Score", "Distance_to_Center"]]


y = df["Price"].values


print(f"Shape of x is: {X.shape}\n")
print(f"Data type of x is: {X.dtypes}\n")

n_samples , n_features = X.shape


coefs = np.ones(n_features + 1)

prediction_bydef = X @ coefs[1:] + coefs[0]


#Appending a column of ones in the X
X = np.hstack((np.ones((n_samples, 1)), X))

predictions = X @ coefs


is_same = np.allclose(prediction_bydef, predictions)


errors = y - predictions

# relative error
rel_err = errors  / y

# Brute force loss calculation
loss_loop = 0
for i in range(n_samples):
    loss_loop = loss_loop + errors[i] ** 2
loss_loop = loss_loop / n_samples

loss_matrix = np.transpose(errors) @ errors / n_samples

is_diff = np.allclose(loss_loop, loss_matrix)
print(f"Are the loss by direct method and matrix calculation: {is_diff}")

print(f"Size of errors: {errors.shape}")
print(f"L2 Norm of errors: {np.linalg.norm(errors)}")
print(f"L2 norm of rel_error: {np.linalg.norm(rel_err)}")


# What is my optimization problem?
# I want to find the coeffs that minimize the mean sqaured error this problem 
# is called as least squares problem


# Objective function: f(coefs) = 1/n_samples * ||y - X @ coefs||^2

# What is the solution?
# A solution is a set of coefficients that minimize the objective function


# Write the loss matrix in terms of the data and the coeffs 

loss_matrix = (y - X @ coefs).T @ (y - X @ coefs) / n_samples

# scalar / vector derivative
grad_matrix = -2/n_samples * X.T @ (X @ coefs - y) / n_samples

# we set the gradient to zero to find the minimum
# -2/n_samples * X.T @ (X @ coefs - y) = 0
# X.T @ X @ coefs = X.T @ y
# coefs = (X.T @ X)^-1 @ X.T @ y

coefs = np.linalg.inv(X.T @ X) @ X.T @ y

# save the coefs in a file
np.savetxt("coeffs.csv", coefs, delimiter=",")

predictions_model = X @ coefs
errors_model = y - predictions_model

print(f"Norm of errors: {np.linalg.norm(errors_model)}")
#Print the L2 norm of the errors
print(f"L2 norm of rel_error: {np.linalg.norm(errors_model / y)}")

#Use all the features in the dataset to build the linear model

X = df.drop("Price", axis=1)
y = df["Price"].values

n_samples, n_features = X.shape
print(f"Number of samples: {n_samples, n_features}")

# Solve the linear model using the Normal Equation
X = np.hstack((np.ones((n_samples, 1)), X))
coefs = np.linalg.inv(X.T @ X) @ X.T @ y


# print the relative error
errors = y - X @ coefs
rel_error = errors / y
print(f"Relative error: {np.linalg.norm(rel_error)}")
# Save coefs in the csv file
np.savetxt("coeffs_all.csv", coefs, delimiter=",")


# calculate the rank of the on X.T @ X
rank = np.linalg.matrix_rank(X.T @ X)
print(f"Rank of X.T @ X: {rank}")

# Solve the normal equation using matrix decomposition

# QR factorization
Q, R = np.linalg.qr(X)
print(f"Shape of Q: {Q.shape}")
print(f"Shape of R: {R.shape}")


# Write R to a file names R.csv
np.savetxt("R.csv", R, delimiter=",")

# R * coeffs = b

# X = QR
# X.T @ X  = `R.T @ Q.T @ Q @ R = R.T @ R
# R*coeffs = Q.T @ y

b = Q.T @ y

#coeffs_qr = np.linalg.inv(R) @ b

# Loop to solve R * coeffs = b using back substitution
coeffs_qr_loop = np.zeros(n_features + 1)

for i in range(n_features, -1, -1):
    coeffs_qr_loop[i] = (b[i] - R[i, i+1:] @ coeffs_qr_loop[i+1:]) / R[i, i]

# save the coeffs in a file
np.savetxt("coeffs_qr.csv", coeffs_qr_loop, delimiter=",")

sol = Q.T @ Q
np.savetxt("sol.csv", sol, delimiter=",")


# Solve the normal equation using SVD
# X = U @ S @ V.T

# Eigen decomposition of a square matrix
# A = V @ D @ V^-1
# A^-1  = V @ D^-1 @ V^-1
# A^-1 = V @ D^-1 @ V.T

# X * coeffs = y
# A = X.T @ X

U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Using U, S, Vt find the coeffs
# A = V @ D @ V.T
# A^-1 = V @ D^-1 @ V.T
# V @ D^-1 @ V.T @ X.T @ y = coeffs

# calculate the coeffs using the eigen decomposition
coeffs_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y

# Save the the coeffs in a file

np.savetxt("coeffs_svd.csv", coeffs_svd, delimiter=",")

# find the predictions using the svd_coeffs and find the error
predictions_svd = X @ coeffs_svd

errors_svd = y - predictions_svd
print(f"Norm of errors using the SVD decomposition: {np.linalg.norm(errors_svd)}")
print(f"Relative error using the SVD decomposition: {np.linalg.norm(errors_svd / y)}")

# Find the inverse of X in the least squares sense
# Pseudo inverse of X

# To complete the pseudo inverse we need to find the inverse of S

### In class code

# write X as a product of U, S, Vt
X_svd = U @ np.diag(S) @ Vt

# Solve for X_svd @ coeffs_svd = y
# Normal Equation: X_svd^T @ X_svd @ coeffs = X_svd^T @ y

# replace X_svd with U @ np.diag(S) @ Vt
#Vt ^T @ np.diag(S)^2 @ Vt @ coeffs = Vt ^ T @ np.diag(S) @ U^T @ y
# np.diag(S)^2 @ Vt @ coeffs = np.diag(S) @ U^T @ y
#coeffs = Vt.T @ np.diag(S)^-1 @ U.T @ y


coeffs_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y # fail if system if rank deficient
coeffs_svd_pinv = np.linalg.pinv(X) @ y #Keeps the threshold to find the inverse


# Save the coeffs in a file
np.savetxt("coeffs_svd_pinv.csv", coeffs_svd_pinv, delimiter=",")
np.savetxt("coeffs_svd.csv", coeffs_svd, delimiter=",")

# Rank deficiency: - Pseudo Inverse, SVD -> Inverse cant be found, so we use the other methods

# X_1 = X[:, 0:1]
# coeffs_1 = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ y
# Plot the data on X[:,1] vs y axis
# Also plot the regression line with only X[:, 0] and X[:, 1] as features

# first make X[:,1] as np.arange between min and max of X[:,1]
# then calculate the predictions using the coefficients
# X_feature = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.01)
# plt.scatter(X[:, 1], y)
# plt.plot(X_feature, X_feature * coeffs_svd[1], color="red")
# plt.xlabel("Square Feet")
# plt.ylabel("Price")
# plt.title("Square Feet vs Price")
# plt.show()
# plt.savefig("Square_Feet_vs_Price.png")

# Use X as only square feet  to build a linear model to predict price
X = df[["Square_Feet"]].values
y = df["Price"].values
# add a column of ones to X
X = np.hstack((np.ones((n_samples, 1)), X))
coeffs_1 = np.linalg.inv(X.T @ X) @ X.T @ y

predictions_1 = X @ coeffs_1
X_feature = np.arange(np.min(X), np.max(X), 0.01)
np.savetxt("coeffs_1.csv", X_feature, delimiter=",")
# add a column of ones to X_feature
X_feature = np.hstack((np.ones((X_feature.shape[0], 1)), X_feature.reshape(-1, 1)))

plt.scatter(X[:, 1], y)
plt.plot(X_feature[:, 1],X_feature @ coeffs_1, color="red")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Square Feet vs Price")
plt.show()
