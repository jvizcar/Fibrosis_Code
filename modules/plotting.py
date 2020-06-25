import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, labels=None, title='', save_path=None, fontsize=18):
    """Plot confusion matrix given a list of true labels and predicted labels.

    Parameters
    ----------
    y_true : list
        list of labels (for binary these could be 0s and 1s)
    y_pred : list
        list of predicted labels
    labels : list (optional)
        list of labels to use for the classes
    title : str (optional)
        title to use for confusion matrix
    save_path : str (optional)
        complete filepath to save confusion matrix to
    fontsize : int (optional)
        fontsize to use

    Return
    ------
    cm : np.array
        confusion matrix in array form, rows are the true labels, columns the predicted labels

    """
    if labels is None:
        # get unique labels from values in y variables
        labels = list(set(y_pred) | set(y_true))
        labels.sort()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred).astype(float)
    sns.heatmap(cm.T, square=True, annot=True, fmt='0.0f', xticklabels=labels,  yticklabels=labels, vmin=0,
                cmap='coolwarm', cbar=False, ax=ax, annot_kws={'size': fontsize})
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.xlabel('True', fontsize=fontsize)
    plt.ylabel('Predicted', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
    return cm
