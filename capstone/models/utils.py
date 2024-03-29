import matplotlib.pyplot as plt
import numpy as np
import itertools as itertools

class Utils():
    """ Utilities """
    
    def __init__(self, debug=False):
        self.debug = debug

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """ Plot a confusion matrix """
        
        if self.debug: print('In plot_confusion_matrix')

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
            print(' ')
        else:
            print('Confusion matrix, without normalization')
            print(' ')
        
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

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    def printTP_FP_TN_FN(self, cnfm):
        """ Print, True Negatives, False Positives, False Negatives, True Positives """
        
        if self.debug: print('In printTP_FP_TN_FN')
        
        # Print true_positives, false_positives, true_negatives, false_negatives
        tn, fp, fn, tp = cnfm.ravel()
        print("True Negatives: ", tn)
        print("False Positives: ", fp)
        print("False Negatives: ", fn)
        print("True Positives: ", tp)
        print(' ')