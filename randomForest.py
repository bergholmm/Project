# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt

# ---------------------------------------------------- #
# Load a CSV file
# ---------------------------------------------------- #

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()

    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# ---------------------------------------------------- #
# Tree functions
# ---------------------------------------------------- #

# Build a decision tree
def build_tree(train, n_features):
    root = get_split(train, n_features)
    split(root, n_features)
    return root

# Create child splits for a node or make terminal
def split(node, n_features):
    left, right = node['groups']
    del(node['groups'])

    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # process left child
    node['left'] = get_split(left, n_features)
    split(node['left'], n_features)

    # process right child
    node['right'] = get_split(right, n_features)
    split(node['right'], n_features)

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Select the best split point for a dataset
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)

    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))

    return gini

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# ---------------------------------------------------- #
# Forest functions
# ---------------------------------------------------- #


# Random Forest Algorithm
def random_forest(train, test, sample_size, n_trees, n_features):
    trees = list()
    out_of_bag = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, n_features)
        trees.append(tree)
        t = filter(lambda x: x not in sample, train)
        out_of_bag.append(t)
    predictions = [forest_predict(trees, out_of_bag, row) for row in train]
    return predictions

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# Make a prediction with a list of bagged trees
def forest_predict(trees, out_of_bag, row):
    predictions = list()
    for i in range(len(trees)):
        if row in out_of_bag[i]:
            predictions.append(predict(trees[i], row))
    return max(set(predictions), key=predictions.count)

# ---------------------------------------------------- #
# Other functions
# ---------------------------------------------------- #

# Split a dataset into train and test
def split_dataset(dataset, p):
    train = list(dataset)
    test = list()
    test_size = len(dataset) * p
    while len(test) < test_size:
        index = randrange(len(train))
        test.append(train.pop(index))

    return test, train

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
                correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate(test_set, train_set, n_features, n_trees, sample_size):
    predicted = random_forest(train_set, test_set, sample_size, n_trees, n_features)
    actual = [row[-1] for row in train_set]
    accuracy = accuracy_metric(actual, predicted)
    return accuracy

# ---------------------------------------------------- #
# OUTER:
# Set aside 10% of the training data
# Create 100 trees with feature selection and bagging with F=1 and F=ln(M+1) (twice)
# Now use the 10% set aside data to get test set error of the two runs and select the lower value
# run 100 times to get converged test error
#
# INNTER:
# Each new training set is drawn, with replacement, from the original training set for each tree
# Then a tree is grown on the new training set using random feature selection. The trees grown are not pruned.
#
# ---------------------------------------------------- #

def main():
    # # Test the random forest algorithm
    # seed(1)

    # load and prepare data
    filename = 'sonar.all-data.csv'
    dataset = load_csv(filename)

    # convert string attributes to integers
    for i in range(0, len(dataset[0])-1):
            str_column_to_float(dataset, i)

    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)

    # evaluate algorithm
    sample_size = 1.0
    n_features = 1
    n_trees = 100

    test, train = split_dataset(dataset, 0.1)
    scores = list()
    # for i in range(100):
    score = evaluate(test, dataset, n_features, n_trees, sample_size)
    scores.append(score)
    #     print(i)

    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

if __name__ == "__main__":
    main()
