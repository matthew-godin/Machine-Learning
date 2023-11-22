# Important note:
# the MNIST dataset must be manually downloaded for this script to work
# with structure mnist/MNIST/raw placed in the root of this folder
import os
import codecs
import numpy
import pickle

mnist_directory_path = "mnist/MNIST/raw/"
mnist_file_paths = os.listdir(mnist_directory_path)
train_data = None
train_labels = None
test_data = None
test_labels = None
gmm_dir = 'gmm/'
pi_path = os.path.join(gmm_dir, 'pi.pkl')
n_path = os.path.join(gmm_dir, 'n.pkl')
counter_n_path = os.path.join(gmm_dir, 'counter_n.pkl')
mu_path = os.path.join(gmm_dir, 'mu.pkl')
counter_mu_path = os.path.join(gmm_dir, 'counter_mu.pkl')
covariance_path = os.path.join(gmm_dir, 'covariance.pkl')
w_path = os.path.join(gmm_dir, 'w.pkl')
w_0_path = os.path.join(gmm_dir, 'w_0.pkl')

def save_list(list_to_save, file_path):
  f = open(file_path, "wb")
  pickle.dump(list_to_save, f)
  f.close()

def load_list(file_path):
  f = open(file_path, "rb")
  loaded_list = pickle.load(f)
  f.close()
  return loaded_list

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

# Creates and saves a Gaussian Mixture Model for given training dataset
def create_and_save_GMM(training_dataset, training_dataset_labels):
    pi = []
    n = []
    counter_n = []
    training_dataset_label_subsets = []
    training_dataset_label_counter_subsets = []
    mu = []
    counter_mu = []
    training_dataset_label_subsets_distances = []
    training_dataset_label_subsets_counter_distances = []
    s = []
    counter_s = []
    covariance = []
    covariance_inverse = []
    w = []
    w_0 = []
    for digit in range(10):
        print(digit)
        pi_i = 0
        for training_dataset_label in training_dataset_labels:
            if training_dataset_label == digit:
                pi_i += 1
        pi.append(pi_i / len(training_dataset_labels))
        digit_indices = []
        digit_counter_indices = []
        for i, training_dataset_label in enumerate(training_dataset_labels):
            if training_dataset_label == digit:
                digit_indices.append(i)
            else:
                digit_counter_indices.append(i)
        n.append(len(digit_indices))
        counter_n.append(len(digit_counter_indices))
        training_dataset_label_subsets.append(numpy.take(training_dataset, digit_indices, 0))
        training_dataset_label_counter_subsets.append(numpy.take(training_dataset, digit_counter_indices, 0))
        subset_point_component_averages = []
        subset_point_component_counter_averages = []
        print(".1")
        # For each component in a point (for each pixel in an image)
        for i in range(len(training_dataset_label_subsets[-1][0])):
            subset_point_component_total = 0
            subset_point_component_counter_total = 0
            # For each point in the subset (for each image in the subset)
            for subset_point in training_dataset_label_subsets[-1]:
                subset_point_component_total += subset_point[i]
            # For each point in the counter subset (for each image in the subset)
            for counter_subset_point in training_dataset_label_counter_subsets[-1]:
                subset_point_component_counter_total += counter_subset_point[i]
            subset_point_component_averages.append(subset_point_component_total
                / len(training_dataset_label_subsets[-1]))
            subset_point_component_counter_averages.append(subset_point_component_counter_total
                / len(training_dataset_label_counter_subsets[-1]))
        mu.append(numpy.array(subset_point_component_averages))
        counter_mu.append(numpy.array(subset_point_component_counter_averages))
        training_dataset_label_subset_distances = []
        training_dataset_label_subset_counter_distances = []
        print(".2")
        # For each point in the subset (for each image in the subset)
        for subset_point in training_dataset_label_subsets[-1]:
            training_dataset_label_point_distances = []
            # For each component in a point (for each pixel in an image)
            for i, subset_point_component in enumerate(subset_point):
                training_dataset_label_point_distances.append(subset_point_component - mu[-1][i])
            training_dataset_label_subset_distances.append(training_dataset_label_point_distances)
        print(".3")
        # For each point in the counter subset (for each image in the subset)
        for counter_subset_point in training_dataset_label_counter_subsets[-1]:
            training_dataset_label_point_counter_distances = []
            # For each component in a point (for each pixel in an image)
            for i, counter_subset_point_component in enumerate(counter_subset_point):
                training_dataset_label_point_counter_distances.append(counter_subset_point_component - counter_mu[-1][i])
            training_dataset_label_subset_counter_distances.append(training_dataset_label_point_counter_distances)
        print(".4")
        training_dataset_label_subsets_distances.append(numpy.array(training_dataset_label_subset_distances))
        training_dataset_label_subsets_counter_distances.append(numpy.array(training_dataset_label_subset_counter_distances))
        s.append(training_dataset_label_subsets_distances[-1].T.dot(training_dataset_label_subsets_distances[-1])
            / len(training_dataset_label_subsets[-1]))
        counter_s.append(training_dataset_label_subsets_counter_distances[-1].T.dot(training_dataset_label_subsets_counter_distances[-1])
            / len(training_dataset_label_counter_subsets[-1]))
        covariance.append(len(training_dataset_label_subsets[-1]) / len(training_dataset) * s[-1]
            + len(training_dataset_label_counter_subsets[-1]) / len(training_dataset) * counter_s[-1])
        # Same as inverse_covariance = numpy.linalg.inv(covariance) (not possible because more than one solution)
        covariance_inverse.append(numpy.linalg.lstsq(covariance[-1], numpy.eye(len(covariance[-1]), len(covariance[-1])), rcond=None)[0])
        w.append(covariance_inverse[-1].dot(mu[-1] - counter_mu[-1]))
        w_0.append(-1 / 2 * mu[-1].T.dot(covariance_inverse).dot(mu[-1])
            + 1 / 2 * counter_mu[-1].T.dot(covariance_inverse).dot(counter_mu[-1])
            + numpy.log(pi[-1] / (1 - pi[-1])))
    save_list(pi, pi_path)
    save_list(n, n_path)
    save_list(counter_n, counter_n_path)
    save_list(mu, mu_path)
    save_list(counter_mu, counter_mu_path)
    save_list(covariance, covariance_path)
    save_list(w, w_path)
    save_list(w_0, w_0_path)

# Loads saved Gaussian Mixture Model and report accuracy for given validation dataset
def load_GMM_and_report_accuracy(validation_dataset, validation_dataset_labels):
    pi = load_list(pi_path)
    n = load_list(n_path)
    counter_n = load_list(counter_n_path)
    mu = load_list(mu_path)
    counter_mu = load_list(counter_mu_path)
    covariance = load_list(covariance_path)
    w = load_list(w_path)
    w_0 = load_list(w_0_path)
    num_accurate = 0
    for i in range(len(validation_dataset)):
        probabilities = []
        for digit in range(10):
            probabilities.append(1 / (1 + numpy.exp(-(w[digit].dot(validation_dataset[i]) + w_0[digit][0]))))
        prediction = probabilities.index(max(probabilities))
        if prediction == validation_dataset_labels[i]:
            num_accurate += 1
    return num_accurate / len(validation_dataset)

# Read and store MNIST data
for mnist_file_path in mnist_file_paths:
    # Disregard gunzip files and only consider extracted files
    if mnist_file_path.endswith('ubyte'):
        with open(mnist_directory_path + mnist_file_path, 'rb') as mnist_file:
            mnist_file_content = mnist_file.read()
            # Magic number indicating which file it is
            mnist_file_type = int(codecs.encode(mnist_file_content[:4], 'hex'), 16)
            # Number of data points in the dataset
            mnist_file_length = int(codecs.encode(mnist_file_content[4:8], 'hex'), 16)
            if mnist_file_type == 2051:
                # Read data points
                read_data = numpy.frombuffer(mnist_file_content, dtype=numpy.uint8, offset=16)
                # Reshape to size num_data_points x (height * width)
                reshaped_data = read_data.reshape(mnist_file_length,
                    int(codecs.encode(mnist_file_content[8:12], 'hex'), 16)
                    * int(codecs.encode(mnist_file_content[12:16], 'hex'), 16))
                if mnist_file_length == 10000:
                    test_data = reshaped_data
                elif mnist_file_length == 60000:
                    train_data = reshaped_data
            elif mnist_file_type == 2049:
                # Read labels
                read_labels = numpy.frombuffer(mnist_file_content, dtype=numpy.uint8, offset=8)
                # Reshape to size num_data_points
                reshaped_labels = read_labels.reshape(mnist_file_length)
                if mnist_file_length == 10000:
                    test_labels = reshaped_labels
                elif mnist_file_length == 60000:
                    train_labels = reshaped_labels
# Create and save GMM for MNIST data
create_and_save_GMM(train_data, train_labels)
accuracy = load_GMM_and_report_accuracy(test_data, test_labels)
print("Accuracy: " + str(accuracy * 100) + "%")
