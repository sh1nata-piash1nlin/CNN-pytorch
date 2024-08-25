# -*- coding: utf-8 -*-
"""
    @author: Nguyen "sh1nata" Duc Tri <tri14102004@gmail.com>
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parse = argparse.ArgumentParser(description='Football Jerseys')
    parse.add_argument('-p', '--data_path', type=str, default='./data/animals/')
    parse.add_argument('-p2', '--data_path2', type=str, default='./data/animals/')
    parse.add_argument('-b', '--batch_size', type=int, default=32)
    parse.add_argument('-e', '--epochs', type=int, default=100)
    parse.add_argument('-l', '--lr', type=float, default=1e-2)
    parse.add_argument('-s', '--image_size', type=int, default=224)
    parse.add_argument('-c', '--checkpoint_path', type=str, default=None) 
    parse.add_argument('-t', '--tensorboard_path', type=str, default="tensorboard")
    parse.add_argument('-r', '--trained_models', type=str, default="trained_models")
    args = parse.parse_args()
    return args


def plot_confusion_matrix(writer, cm, class_names, epoch):

    figure = plt.figure()
    #color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="plasma")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    #Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    #Use white text if squares are dark, otherwise black
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('Confusion Matrix', figure, epoch)
