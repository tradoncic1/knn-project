import matplotlib.pyplot as plt
import numpy as np
import struct
import math
from PIL import Image
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class UTILS :

    def read_idx(filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))

            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

    def getMatrix(array):
        matrices = []

        for item in array:
            matrix = np.reshape(item, (28, 28))
            matrices.append(matrix)

        return matrices


    def showImage(array, size):
        images = []
        for i in range(size):
            images.append(Image.fromarray(array[i]))

        width = 28 * size
        height = 28
        if size > 20:
            width = 20 * 28
            height = math.ceil(size / 20) * 28

        new_image = Image.new('RGB', (width, height))
        y_offset = 0

        for i in range(math.ceil(size / 20)):
            x_offset = 0
            for j in range(20):
                try:
                    new_image.paste(images[i * 20 + j], (x_offset, y_offset))
                    x_offset += 28
                except:
                    break
            y_offset += 28

        new_image.save('test_data.png')
        new_image.show()



    def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true)]
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax
