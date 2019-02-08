import itertools
import os
import re
import shutil
from textwrap import wrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import torch
import utils.evaluator as eu
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

plt.switch_backend('agg')
plt.axis('scaled')


# TODO: Add custom phase names
class LogWriter(object):
    def __init__(self, num_class, log_dir_name, exp_name, use_last_checkpoint=False, labels=None,
                 cm_cmap=plt.cm.Blues):
        self.num_class = num_class
        train_log_path, val_log_path = os.path.join(log_dir_name, exp_name, "train"), os.path.join(log_dir_name,
                                                                                                   exp_name,
                                                                                                   "val")
        if not use_last_checkpoint:
            if os.path.exists(train_log_path):
                shutil.rmtree(train_log_path)
            if os.path.exists(val_log_path):
                shutil.rmtree(val_log_path)

        self.writer = {
            'train': SummaryWriter(log_dir=train_log_path, comment='Train Summary', flush_secs=30),
            'val': SummaryWriter(log_dir=val_log_path, comment='Val Summary', flush_secs=30)
        }
        self.curr_iter = 1
        self.cm_cmap = cm_cmap
        self.labels = self.beautify_labels(labels)
        self.logger = logging.getLogger()
        file_handler = logging.FileHandler("{0}/{1}.log".format(os.path.join(log_dir_name, exp_name), "console_logs"))
        # console_handler = logging.StreamHandler()
        self.logger.addHandler(file_handler)
        # self.logger.addHandler(console_handler)

    def log(self, text, phase='train'):
        self.logger.info(text)

    def loss_per_iter(self, loss_value, i_batch, current_iteration):
        self.log('[Iteration : ' + str(i_batch) + '] Loss -> ' + str(loss_value))
        self.writer['train'].add_scalar('loss/per_iteration', loss_value, current_iteration)

    def loss_per_epoch(self, loss_arr, phase, epoch):
        if phase == 'train':
            loss = loss_arr[-1]
        else:
            loss = np.mean(loss_arr)
        self.writer[phase].add_scalar('loss/per_epoch', loss, epoch)
        self.log('epoch ' + phase + ' loss = ' + str(loss))

        return loss

    def cm_per_epoch(self, phase, output, correct_labels, epoch):

        self.log("Confusion Matrix...")
        _, cm = eu.dice_confusion_matrix(output, correct_labels, self.num_class, mode='train')
        self.plot_cm('confusion_matrix', phase, cm, epoch)
        self.log("DONE")

    def plot_cm(self, caption, phase, cm, step=None):
        fig = matplotlib.figure.Figure(figsize=(8, 8), dpi=180, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(cm, interpolation='nearest', cmap=self.cm_cmap)
        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_xticks(np.arange(self.num_class))
        c = ax.set_xticklabels(self.labels, fontsize=4, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=7)
        ax.set_yticks(np.arange(self.num_class))
        ax.set_yticklabels(self.labels, fontsize=4, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], '.2f') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=6,
                    verticalalignment='center', color="white" if cm[i, j] > thresh else "black")

        fig.set_tight_layout(True)
        np.set_printoptions(precision=2)
        if step:
            self.writer[phase].add_figure(caption + '/' + phase, fig, step)
        else:
            self.writer[phase].add_figure(caption + '/' + phase, fig)

    def dice_score_per_epoch(self, phase, output, correct_labels, epoch):
        self.log("Dice Score...")

        # TODO: multiclass vs binary
        ds = eu.dice_score_binary(output, correct_labels, self.num_class, phase)
        self.log('Dice score is ' + str(ds))
        # self.plot_dice_score(phase, 'dice_score_per_epoch', ds, 'Dice Score', epoch)

        self.log("DONE")
        return ds

    def dice_score_per_epoch_segmentor(self, phase, output, correct_labels, epoch):
        self.log("Dice Score...")

        # TODO: multiclass vs binary
        ds = eu.dice_score_perclass(output, correct_labels, self.num_class, mode=phase)
        ds_mean = torch.mean(ds[1:])
        self.log('Dice score is ' + str(ds))
        self.log('Dice score mean ' + str(ds_mean))
        self.plot_dice_score(phase, 'dice_score_per_epoch', ds, 'Dice Score', epoch)
        self.log("DONE")
        return ds_mean

    def plot_dice_score(self, phase, caption, ds, title, step=None):
        fig = matplotlib.figure.Figure(figsize=(8, 6), dpi=180, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(title, fontsize=10)
        ax.xaxis.set_label_position('top')
        ax.bar(np.arange(self.num_class), ds)
        ax.set_xticks(np.arange(self.num_class))
        c = ax.set_xticklabels(self.labels, fontsize=6, rotation=-90, ha='center')
        ax.xaxis.tick_bottom()
        if step:
            self.writer[phase].add_figure(caption + '/' + phase, fig, step)
        else:
            self.writer[phase].add_figure(caption + '/' + phase, fig)

    def plot_eval_box_plot(self, caption, class_dist, title):
        fig = matplotlib.figure.Figure(figsize=(8, 6), dpi=180, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(title, fontsize=10)
        ax.xaxis.set_label_position('top')
        ax.boxplot(class_dist)
        ax.set_xticks(np.arange(self.num_class))
        c = ax.set_xticklabels(self.labels, fontsize=6, rotation=-90, ha='center')
        ax.xaxis.tick_bottom()
        self.writer['val'].add_figure(caption, fig)

    def image_per_epoch_segmentor(self, prediction, ground_truth, phase, epoch):
        self.log("Sample Images...")
        ncols = 2
        nrows = len(prediction)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 20))

        for i in range(nrows):
            ax[i][0].imshow(torch.squeeze(ground_truth[i]), cmap='CMRmap', vmin=0, vmax=self.num_class - 1)
            ax[i][0].set_title("Ground Truth", fontsize=10, color="blue")
            ax[i][0].axis('off')
            ax[i][1].imshow(torch.squeeze(prediction[i]), cmap='CMRmap', vmin=0, vmax=self.num_class - 1)
            ax[i][1].set_title("Predicted", fontsize=10, color="blue")
            ax[i][1].axis('off')
        fig.set_tight_layout(True)
        self.writer[phase].add_figure('sample_prediction/' + phase, fig, epoch)
        self.log('DONE')

    def image_per_epoch(self, prediction, ground_truth, phase, epoch, additional_image=None):
        self.log("Sample Images...")
        ncols = 3
        nrows = len(prediction)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 20))

        for i in range(nrows):
            ax[i][0].imshow(additional_image[i].squeeze(), cmap='gray', vmin=0, vmax=5)
            ax[i][0].set_title("Input Image", fontsize=10, color="blue")
            ax[i][0].axis('off')
            ax[i][1].imshow(ground_truth[i].squeeze(), cmap='jet', vmin=0, vmax=5)
            ax[i][1].set_title("Ground Truth", fontsize=10, color="blue")
            ax[i][1].axis('off')
            ax[i][2].imshow(prediction[i].squeeze(), cmap='jet', vmin=0, vmax=5)
            ax[i][2].set_title("Predicted", fontsize=10, color="blue")
            ax[i][2].axis('off')
        fig.set_tight_layout(True)
        self.writer[phase].add_figure('sample_prediction/' + phase, fig, epoch)
        self.log('DONE')

    def graph(self, model, X):
        self.writer['train'].add_graph(model, X)

    def close(self):
        self.writer['train'].close()
        self.writer['val'].close()

    def beautify_labels(self, labels):
        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
        classes = ['\n'.join(wrap(l, 40)) for l in classes]
        return classes
