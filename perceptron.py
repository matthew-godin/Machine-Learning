# Python Version: 3.8.10
import numpy
import matplotlib.pyplot as plt

# Constants
n = 4601
d = 57
max_pass = 500

# Initialize the hyperplane
w = numpy.array([0.0] * d)
b = 0

# Read data from spambase.data
spambase = open('spambase.data', 'r')
spambase_lines = spambase.readlines()
spambase.close()

# Convert Spambase data to Numpy matrices X and Y
X_vals = [[0.0] * d] * n
Y_vals = [0] * n
for i, spambase_line in enumerate(spambase_lines):
    spambase_line_string_vals = spambase_line.split(',')
    vals = [float(string_val) for string_val in spambase_line_string_vals]
    for j, val in enumerate(vals[:-1]):
        X_vals[i][j] = val
    Y_vals[i] = -1 if vals[-1] == 1 else 1
X = numpy.array(X_vals)
Y = numpy.array(Y_vals)

# Initialize the count of mistakes per pass
mistake = [0] * max_pass

# Perform the perceptron algorithm
for t in range(max_pass):
    for i, x in enumerate(X):
        y = Y[i]
        if numpy.dot(x, w) + b <= 0:
            w += y * x
            b += y
            mistake[t] += 1

# Plot the number of mistakes with respect
# to the number of passes using Matplotlib
num_passes = range(1, max_pass + 1)
plt.plot(num_passes, mistake)
plt.xlabel('Number of Passes')
plt.ylabel('Number of Mistakes')
plt.title('Number of Mistakes With Respect to Number of Passes for Spambase Perceptron')
plt.show()