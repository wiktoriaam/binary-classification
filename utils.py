import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns   


# function to split data where the train set is some fraction of the whole set, defined by train_fraction
# assumes there are two classes, takes their initial values as initial_class_values and changes them to [0,1]
def split_data(data, train_fraction, initial_class_values=[2,4]):
    
    data_copy = data.copy()

    np.random.shuffle(data_copy)
    X = data_copy[:, :-1]
    Y = data_copy[:, -1]

    Y[Y == initial_class_values[0]] = 0
    Y[Y == initial_class_values[1]] = 1

    class_0 = [i for i, c in enumerate(Y) if c == 0]
    class_1 = [i for i, c in enumerate(Y) if c == 1]

    train_idx = class_0[:int(len(class_0)*train_fraction)] + class_1[:int(len(class_1)*train_fraction)]
    test_idx = class_0[int(len(class_0)*train_fraction):] + class_1[int(len(class_1)*train_fraction):]

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    X_train = X[train_idx]
    X_test = X[test_idx]
    Y_train = Y[train_idx].reshape(-1,1)
    Y_test = Y[test_idx].reshape(-1,1)

    return X_train, Y_train, X_test, Y_test



# function that takes different models, each with one set of parameters given, 
# and returns average error rate on each of given fractions of training set
def calculate_average_error(fractions, data, models, iterations, parameters):
    results = np.zeros((len(models), len(fractions), iterations))

    for i in range(iterations):
        X_train, Y_train, X_test, Y_test = split_data(data, 2/3)
        class_0 = [i for i, c in enumerate(Y_train) if c == 0]
        class_1 = [i for i, c in enumerate(Y_train) if c == 1]
        samples0 = len(class_0)
        samples1 = len(class_1)

        for f in range(len(fractions)):
            X_subset = np.concatenate((X_train[class_0[:int(fractions[f] * samples0)]], X_train[class_1[:int(fractions[f] * samples1)]]))
            Y_subset = np.concatenate((Y_train[class_0[:int(fractions[f] * samples0)]], Y_train[class_1[:int(fractions[f] * samples1)]]))
            
            for m in range(len(models)):
                model = models[m]
                model.fit(X_subset, Y_subset, *parameters[m])
                Y_pred = model.predict(X_test)
                results[m][f][i] = model.error(Y_test, Y_pred)
    print(results.shape)

    return np.mean(results, axis=2)







# returns the confusion matrix for Y and predictions
def confusion_matrix(Y, Y_pred):
    result = np.zeros((2,2))

    for i in range(len(Y)):
        result[int(Y_pred[i][0])][int(Y[i][0])] += 1

    return result

# plots the confusion matrix for Y and its prediction
def plot_confusion_matrix(Y, Y_pred, title):
    matrix = confusion_matrix(Y, Y_pred)

    f, ax = plt.subplots(figsize=(5,3))
    ax = sns.heatmap(matrix.astype(int), annot=True, fmt="d", cmap = "OrRd")

    ax.set_xlabel("Real values")
    ax.set_ylabel("Predicted values")
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    
    plt.title(title)
    plt.show()




# accepts one model and different sets of parameters
# returns average error rates on test set after learning on different fractions of training set for each set of parameters
# performs iterations on different train and test sets to calculate the average error
def get_error_rates(data, model, parameter_sets, fractions, iterations, train_set_size):
    result = np.zeros((len(parameter_sets), len(fractions), iterations))

    for i in range(iterations):
        X_train, Y_train, X_test, Y_test = split_data(data, train_set_size)
        class_0 = [i for i, c in enumerate(Y_train) if c == 0]
        class_1 = [i for i, c in enumerate(Y_train) if c == 1]
        samples0 = len(class_0)
        samples1 = len(class_1)
        for f in range(len(fractions)):
            X_subset = np.concatenate((X_train[class_0[:int(fractions[f] * samples0)]], X_train[class_1[:int(fractions[f] * samples1)]]))
            Y_subset = np.concatenate((Y_train[class_0[:int(fractions[f] * samples0)]], Y_train[class_1[:int(fractions[f] * samples1)]]))
            
            for p in range(len(parameter_sets)):
                model.fit(X_subset, Y_subset, *parameter_sets[p])
                Y_pred = model.predict(X_test)
                result[p][f][i] = model.error(Y_test, Y_pred)

    result = np.mean(result, axis = 2)

    return result
            
# plots learning curves given fractions of the training set and average test set error rates on these fractions
def plot_learning_curves(error_rates, xlegend, xlabel, ylabel, title, fractions, legend_title):
    plt.figure(figsize=(7,5))
    sns.set_style("whitegrid")

    for error in range(len(error_rates)):
        plt.plot(fractions, error_rates[error], label = xlegend[error], marker='o')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title = legend_title)
    plt.show()

    return error_rates[:,-1]



