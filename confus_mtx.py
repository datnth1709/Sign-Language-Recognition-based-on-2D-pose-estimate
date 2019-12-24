import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import plot_confusion_matrix

y_true = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
                    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 
                    ])
y_pred = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2, 2, 0, 2, 2,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                    2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 
                    5, 5, 5, 5, 5, 5, 1, 5, 5, 5,
                    1, 6, 6, 6, 3, 6, 3, 6, 6, 6,
                    5, 0, 0, 0, 0, 0, 0, 7, 0, 7,
                    8, 8, 8, 8, 8, 3, 0, 0, 8, 0, 
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                    ])
cnf_matrix = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.BuGn):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class_names = ["Stand","Sit","Talk","Carry/Hold","Answer phone","Lie/Sleep","Fight","Fall Down","Knee/Crouch","Walk"]
#                0      1      2       3                4           5          6         7          8           9          
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Ma trận nhầm lẫn')

plt.show()
