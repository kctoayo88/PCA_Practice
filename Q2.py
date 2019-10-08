# ##Import required module
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ==========================Q2==================================
# Use testing data to predict its class.
def Q2_test_data(testing_data):
  y1_list = []
  for i in range(len(testing_data)):
    y1 = np.dot(Q2_E1, testing_data[i])
    y1_list.append(y1)
  return y1_list

# Use testing data to predict its class.
def test_data(testing_data):
  result_list = []
  for i in range(len(testing_data)):
    y1 = np.dot(E1, testing_data[i])
    dist_y1 = np.linalg.norm(y1 - Normal_y_mean)
    y2 = np.dot(E2, testing_data[i])
    dist_y2 = np.linalg.norm(y2 - Cyclic_y_mean)
    y3 = np.dot(E3, testing_data[i])
    dist_y3 = np.linalg.norm(y3 - Up_shift_y_mean)

    delta_y = np.array([dist_y1, dist_y2, dist_y3])
    which_class = np.argmin(delta_y)
    result_list.append(which_class)
  return result_list

# Get Eigenvector
def get_Ek(training_data):
  # ## Transpose matrix
  Transpose = training_data.T  # Transpose matrixmal

  # ## Covariance matrix
  cov = np.cov(Transpose)  # covariance matrix

  # ## Eigenvalue and Eigenvector
  Eigenvalues, Eigenvector = np.linalg.eig(cov)  # Eigenvector
  Ek = np.array(Eigenvector[0:K])
  return Ek

# Get Y mean
def get_y_mean(Ek, training_data):
  # ## Matrix multiplication
  y_list = []

  # Compute the all of y.
  for i in range(len(training_data)):
      X = training_data[i]
      y = np.dot(Ek, X)  # matrix y*matrix Eigenvector
      y_list.append(y)
  
  # Get mean
  y_list = np.array(y_list)
  y_mean = np.mean(y_list, axis=0)
  return y_mean

# ## Load data
data_homework_1 = np.loadtxt(open("./f1.csv","rb"),delimiter=",",skiprows=0)
data_homework_2 = np.loadtxt(open("./f2.csv","rb"),delimiter=",",skiprows=0)

# Training
# ## Divide the data into training and testing
Normal_signal_train = data_homework_1[0:50, :]
Normal_signal_test = data_homework_1[50:100, :]
Cyclic_signal_train = data_homework_1[100:150, :]
Cyclic_signal_test = data_homework_1[150:200, :]
Up_shift_signal_train = data_homework_1[200:250, :]
Up_shift_signal_test = data_homework_1[250:300, :]

#Set up the value of K
K = 10

# Get E1, E1, E2 parameters
E1 = get_Ek(Normal_signal_train)
E2 = get_Ek(Cyclic_signal_train)
E3 = get_Ek(Up_shift_signal_train)

# Get the means of each class
Normal_y_mean = get_y_mean(E1, Normal_signal_train)
Cyclic_y_mean = get_y_mean(E2, Cyclic_signal_train)
Up_shift_y_mean = get_y_mean(E3, Up_shift_signal_train)

# 2a
# Testing
C1_prediction = test_data(Normal_signal_test)
C2_prediction = test_data(Cyclic_signal_test)
C3_prediction = test_data(Up_shift_signal_test)

# Combine all prediction result together
prediction_list = []
prediction_list.extend(C1_prediction)
prediction_list.extend(C2_prediction)
prediction_list.extend(C3_prediction)

# Create the label list (A known class)
label_list = []
for i in range(50):
  label_list.append(0)
for i in range(50, 100):
  label_list.append(1)
for i in range(100, 150):
  label_list.append(2)

# Show as a confusion matrix 
c_matrix = confusion_matrix(label_list, prediction_list)
print(c_matrix)

#2b
K = 2

# Get E1, E1, E2 parameters
Q2_E1 = get_Ek(Normal_signal_train)

C1_result = np.array(Q2_test_data(Normal_signal_test))
C2_result = np.array(Q2_test_data(Cyclic_signal_test))
C3_result = np.array(Q2_test_data(Up_shift_signal_test))

# Plot the y of Class 1, Class 2 and Class3
plt.scatter(C1_result[0:100,0], C1_result[0:100,1], color='red', label='Normal')
plt.scatter(C2_result[0:100,0], C2_result[0:100,1], color='green', label='Cyclic')
plt.scatter(C3_result[0:100,0], C3_result[0:100,1], color='blue', label='Upward')
plt.legend(loc='upper right')
plt.savefig("./homework_Q2.png", dpi=300, format="png")
plt.show()
