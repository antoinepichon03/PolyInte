import numpy as np
import matplotlib.pyplot as plt
from q3_1 import gradient_descent_ridge
from q1_1 import rmse


eta0 = 0.001
k = 0.001
T = 100
lambdaa = 1.0
schedules = ["constant", "exp_decay", "cosine"]
results = []

X_test = np.loadtxt("X_test.csv", delimiter=",",  skiprows=1)
y_test = np.loadtxt("y_test.csv", delimiter=",",  skiprows=1)
X_train = np.loadtxt("X_train.csv", delimiter=",",  skiprows=1)
y_train = np.loadtxt("y_train.csv", delimiter=",",  skiprows=1)


plt.figure()
for schedule in schedules:
    w , losses = gradient_descent_ridge(X_train, y_train , lambdaa , eta0, T, schedule, k)
    plt.plot(range(T), losses, label=schedule)
    y_hat = X_test @ w
    Rmse = rmse(y_test, y_hat)
    print(f"RMSE for {schedule}:{Rmse}")

plt.xlabel("Iteration")
plt.ylabel("Training Loss") 
plt.title("Training Loss for defferent Learning rate shedules")
plt.grid()
plt.legend()
plt.show()