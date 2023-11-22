# Python Version: 3.8.10
import numpy
import math
import matplotlib.pyplot as plt

# Constants
logistic_regression_dataset_path = 'a2_files/logistic_regression-dataset/'
d = 64
test_n = 110
train_n = 100
num_train_datasets = 10
num_newton_iterations = 10

# Returns list of lines of file at given path
def get_file_lines(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    return lines

# Returns Numpy matrix for given lines of comma separated values
def get_numpy_matrix(lines):
    vals = []
    for line in lines:
        if ',' not in line:
            vals.append(1 if line.startswith('6') else -1)
            continue
        line_string_vals = line.split(',')
        line_vals = [int(line_string_val) for line_string_val in line_string_vals]
        row_vals = []
        for line_val in line_vals:
            row_vals.append(line_val)
        vals.append(row_vals)
    return numpy.array(vals)

# Computes the sigmoid od the given value
def sigmoid(val):
    return 1 / (1 + math.exp(-val))

# Computes probabilities for each dataset that's it belongs to the first class using
# the sigmoid function of the inner product of the dataset and w (weight) plus w_0 (weight_intercept)
def get_probabilities(data, weight, weight_intercept):
    probability_vals = []
    for dataset in data:
        probability_vals.append(sigmoid(dataset.dot(weight) + weight_intercept))
    return numpy.array(probability_vals)

# Computes gradient resulting from each probability, each label, and each dataset,
# regularizing using the given lambda (hyperparam) and w (weight)
def get_gradient(data, labels, probs, hyperparam, weight):
    probability_vals = []
    for i in range(len(labels)):
        probability_vals.append(probs[i] - (1 + labels[i]) / 2)
    probability_input = numpy.array(probability_vals)
    data_columns = data.T
    gradient_vals = []
    for i in range(len(weight)):
        gradient_vals.append(data_columns[i].dot(probability_input) + hyperparam * weight[i])
    return numpy.array(gradient_vals)

# Computes inverse matrix of hessian resulting from each probability and each dataset,
# regularizing using the given lambda (hyperparam)
def get_inverse_hessian(data, probs, hyperparam):
    inverse_probability_vals = []
    for prob in probs:
        inverse_probability_vals.append(1 - prob)
    inverse_probs = numpy.array(inverse_probability_vals)
    prob_odds = numpy.multiply(probs, inverse_probs)
    data_columns = data.T
    hessian_left_hand_vals = []
    for data_column in data_columns:
        hessian_left_hand_vals.append(numpy.multiply(prob_odds, data_column))
    hessian_left_hand = numpy.array(hessian_left_hand_vals)
    return numpy.linalg.inv(numpy.matmul(hessian_left_hand, data) + hyperparam * numpy.eye(len(data_columns)))

# Generate predicted labels for each datasets using the given trained model
def get_predicted_labels(data, weight, weight_intercept):
    predicted_labels_vals = []
    for dataset in data:
        predicted_labels_vals.append(1 if sigmoid(dataset.dot(weight) + weight_intercept) > 0.5 else -1)
    return numpy.array(predicted_labels_vals)

# Compute the accuracy of predicted labels against actual labels
def get_accuracy(labels, actual_labels):
    num_accurate = 0
    for i in range(len(labels)):
        if labels[i] == actual_labels[i]:
            num_accurate += 1
    return num_accurate / len(labels)

# Trains a model with given training dataset and returns accuracy against validation dataset
def train_and_get_accuracy(training_dataset, training_dataset_labels,
    validation_dataset, validation_dataset_labels, hyperparam, num_newton_iters,
    print_w_and_w_0=False):
    # Learn w and w_0 by conditional likelihood maximization using Newton's algorithm to optimize the parameters
    # for the given cross-validation training dataset
    w = numpy.zeros(len(training_dataset[0]))
    w_0 = 0.0
    # Perform Newton's algorithm for 10 iterations
    for i in range(num_newton_iters):
        probabilities = get_probabilities(training_dataset, w, w_0)
        gradient = get_gradient(training_dataset, training_dataset_labels, probabilities, hyperparam, w)
        inverse_hessian = get_inverse_hessian(training_dataset, probabilities, hyperparam)
        # Update w after Newton algorithm iteration training
        for i in range(len(w)):
            w[i] -= inverse_hessian[i].dot(gradient)
    # Predict the labels corresponding to the validation datasets
    predicted_labels = get_predicted_labels(validation_dataset, w, w_0)
    if print_w_and_w_0:
        print('w: ')
        print(w)
        print('w_0: ')
        print(w_0)
    # Compute accuracy of the predicted labels
    return get_accuracy(predicted_labels, validation_dataset_labels)

# Read data from logistic_regression-dataset
test_data_csv_lines = get_file_lines(logistic_regression_dataset_path + 'testData.csv')
test_labels_csv_lines = get_file_lines(logistic_regression_dataset_path + 'testLabels.csv')
train_data_csv_lines = []
train_labels_csv_lines = []
for i in range(num_train_datasets):
    train_data_csv_lines.append(get_file_lines(logistic_regression_dataset_path + 'trainData' + str(i + 1) + '.csv'))
    train_labels_csv_lines.append(get_file_lines(logistic_regression_dataset_path + 'trainLabels' + str(i + 1) + '.csv'))

# Convert logistic_regression-dataset data to Numpy matrices test_data, test_labels, train_data[1..10], and train_labels[1..10]
test_data = get_numpy_matrix(test_data_csv_lines)
test_labels = get_numpy_matrix(test_labels_csv_lines)
train_data = []
train_labels = []
for i in range(num_train_datasets):
    train_data.append(get_numpy_matrix(train_data_csv_lines[i]))
    train_labels.append(get_numpy_matrix(train_labels_csv_lines[i]))

# ------------------------------------------------------------------------------------
# 1. Graphs That Show the Cross-Validation Accuracy of Logistic Regression as λ Varies
# ------------------------------------------------------------------------------------

# Try different hyperparameters
hyperparameters = []

# Below code is for graphs 1 to 5
# max_first_digit_excluded = 10 # for λ 1 to 0.9 or 9000000000
# initial_val = 0.0000000001 # for λ 0.0000000001 to 0.9
# initial_val = 1.0 # for λ 1 to 9000000000, 2000000000, or 1000000000
# initial_val = 10000000000.0 # for λ 10000000000 to 90000000000000000000
# initial_val = 100000000000000000000.0 # for λ 100000000000000000000 to 900000000000000000000000000000
# initial_val = 1000000000000000000000000000000.0 # for λ 1000000000000000000000000000000 to 9000000000000000000000000000000000000000
# next_val = initial_val
# max_first_digit_excluded = 3 # for λ 1 to 2000000000
# max_first_digit_excluded = 2 # for λ 1 to 1000000000
# below code is for the above λs
# for i in range(10):
    # for j in range(1, 10):
        # if j == max_first_digit_excluded and i == 9:
            # break
        # hyperparameters.append(initial_val * (10 ** i) * j)
        # next_val = initial_val * (10 ** i) * j

# Below code is for graphs 6 to 13
# initial_val = 0.1 # for λ 0.1 to 2000000000
# order_of_magnitude_excluded = 9 # for λ 0.1 to 2000000000
# order_of_magnitude_excluded = 8 # for λ 0.1 to 250000000
# order_of_magnitude_excluded = 7 # for λ 0.1 to 50000000
# order_of_magnitude_excluded = 6 # for λ 0.1 to 10000000, 2000000
# order_of_magnitude_excluded = 5 # for λ 0.1 to 250000
# order_of_magnitude_excluded = 3 # for λ 0.1 to 10000
# below code is for the above λs
# for i in range(order_of_magnitude_excluded):
    # hyperparameters.append(initial_val * (10 ** i))
# next_val = 0.0 # for λ 0.1 to 2000000000, 250000000, 50000000
# next_val_increment = 20000000 # for λ 0.1 to 2000000000
# next_val_increment = 2500000 # for λ 0.1 to 250000000
# next_val_increment = 500000 # for λ 0.1 to 50000000
# next_val_increment = 100000 # for λ 0.1 to 10000000
# next_val_increment = 20000 # for λ 0.1 to 2000000
# next_val_increment = 2500 # for λ 0.1 to 250000
# next_val_increment = 500 # for λ 0.1 to 50000
# next_val_increment = 100 # for λ 0.1 to 10000
# below code is for the above λs
# for i in range(100):
    # next_val += next_val_increment
    # hyperparameters.append(next_val)

# Below code is for graph 14
initial_val = 6000 # for λ 6000 to 10000
next_val = initial_val
# next_val_increment = 1 # for λ 6000 to 10000, takes ~ 1 hour and 30 minutes (4000 λs)
next_val_increment = 100 # for λ 6000 to 10000, express version, takes less than 1 minute (40 λs)
# num_increments = 4000 # for λ 6000 to 10000, takes ~ 1 hour and 30 minutes (4000 λs)
num_increments = 40 # for λ 6000 to 10000, express version, takes less than 1 minute (40 λs)
# below code is for the above λs
hyperparameters.append(next_val)
for i in range(num_increments):
    next_val += next_val_increment
    hyperparameters.append(next_val)

average_accuracies = []
for hyperparameter in hyperparameters:
    # To know precise values, disabled by default
    # print(hyperparameter)
    # Perform 10-fold cross-validation for different hyperparameters
    # and compute average accuracy throughout the 10-fold cross-validation
    # for the given hyperparameter
    accuracies = []
    for i in range(num_train_datasets):
        validation_data = train_data[i]
        validation_labels = train_labels[i]
        j = 1 if i == 0 else 0
        training_data = train_data[j]
        training_labels = train_labels[j]
        while j < num_train_datasets - 1:
            j += 1
            if j == i:
                continue
            training_data = numpy.concatenate((training_data, train_data[j]))
            training_labels = numpy.concatenate((training_labels, train_labels[j]))
        accuracies.append(train_and_get_accuracy(training_data, training_labels,
            validation_data, validation_labels, hyperparameter, num_newton_iterations))
    average_accuracy = numpy.mean(accuracies)
    # To know precise values, disabled by default
    # print(average_accuracy)
    average_accuracies.append(average_accuracy)
plt.plot(hyperparameters, average_accuracies)
plt.xlabel('λ')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy of Logistic Regression as λ Varies From '
    + numpy.format_float_positional(initial_val, trim='-') + ' to '
    + numpy.format_float_positional(next_val, trim='-')
    + ' (Maximum Accuracy of ' + str(max(average_accuracies)) + ' when λ = '
    + numpy.format_float_positional(hyperparameters[numpy.argmax(average_accuracies)], trim='-') + ')')
plt.show()
# ------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# 2. Accuracy of Logistic Regression on the Test Set With the Best λ for Regularization
# 3. Parameters w and w_0 Found for Logistic Regression
# -------------------------------------------------------------------------------------

best_hyperparameter = 8000.0

i = 0
training_data = train_data[i]
training_labels = train_labels[i]
while i < num_train_datasets - 1:
    i += 1
    training_data = numpy.concatenate((training_data, train_data[i]))
    training_labels = numpy.concatenate((training_labels, train_labels[i]))
best_hyperparameter_accuracy = train_and_get_accuracy(training_data, training_labels,
    test_data, test_labels, best_hyperparameter, num_newton_iterations, True)

print('Accuracy of Logistic Regression on the Test Set With the Best λ for Regularization: '
    + str(best_hyperparameter_accuracy))

# -------------------------------------------------------------------------------------
