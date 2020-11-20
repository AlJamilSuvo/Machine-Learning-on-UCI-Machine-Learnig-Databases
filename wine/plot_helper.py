import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score



def plt_accuracy_learning_curve(history,full_limit=False):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    if full_limit:
        plt.ylim([0,1])
    plt.show()


def plt_loss_learning_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.show()


def plot_confusion_matrix(true_classes,predicted_class,labels):
    cm=confusion_matrix(true_classes,predicted_class)
    ax=plt.subplot()
    group_count=["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentage=["{0:.2f}".format(value) for value in cm.flatten()/np.sum(cm)]
    annot=[f"{v1}\n{v2}%" for v1,v2 in zip(group_count,group_percentage)]
    annot=np.asarray(annot).reshape(cm.shape[0],cm.shape[1])
    sns.heatmap(cm,annot=annot,fmt='',cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)