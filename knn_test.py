import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import pandas as pd
import seaborn as sns

from knn import KNN
from utils import UTILS as utils

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def mnistDataset(number1=1, number2=2, number3=3, test_size=500):
    number1 = int(number1)
    number2 = int(number2)
    number3 = int(number3)
    test_size = int(test_size)

    raw_train = utils.read_idx("train-images-idx3-ubyte")
    X_train = np.reshape(raw_train, (60000, 28*28))
    y_train = utils.read_idx("train-labels-idx1-ubyte")
    idx = (y_train == number1) | (y_train == number2) | (y_train == number3)
    X = X_train[idx]
    Y = y_train[idx]

    raw_test = utils.read_idx("t10k-images-idx3-ubyte")
    X_test = np.reshape(raw_test, (10000, 28*28))
    y_test = utils.read_idx("t10k-labels-idx1-ubyte")
    idx = (y_test == number1) | (y_test == number2) | (y_test == number3)
    x_test = X_test[idx]
    y_true = y_test[idx]


    matrix = utils.getMatrix(x_test[0: test_size])

    if test_size <= 500:
        utils.showImage(matrix, len(matrix))

    return X, x_test[0:test_size], Y, y_true[0:test_size]

def sklearnDataset():
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    return train_test_split(X, y, test_size=0.2, random_state=1234)


#X_train, X_test, y_train, y_test = mnistDataset(3, 8, 4, 20)
X_train, X_test, y_train, y_test = sklearnDataset()

clf = KNN(k=3)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print('searches:\n', y_test)
print('predictions:\n', predictions)
print(acc)

cm = confusion_matrix(y_test, predictions, normalize='true')

# Transform to df for easier plotting
#cm_df = pd.DataFrame(cm, index = ['3','8','4'], columns = ['3','8','4'])
cm_df = pd.DataFrame(cm)

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, cmap="YlGnBu", annot=True)
plt.title('Confusion Matrix \nAccuracy:{0:.3f}'.format(acc))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
#cm = confusion_matrix(y_true, y_pred, labels=["3", "8", "4"])


print('Finished')
