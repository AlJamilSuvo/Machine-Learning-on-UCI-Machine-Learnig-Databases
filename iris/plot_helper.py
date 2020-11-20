import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.metrics import confusion_matrix



def plt_accuracy_learning_curve(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.ylim()
    plt.show()


def plt_loss_learning_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.show()


def plot_confidence_matrix(true_classes,predicted_class,labels):
    cm=confusion_matrix(true_classes,predicted_class)
    ax=plt.subplot()
    sns.heatmap(cm,annot=True,ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)