import os
import numpy as np
import torch
from nn_additional_losses import losses
from torch.optim import lr_scheduler
import torch.nn as nn
from utils.log_utils import LogWriter
import torch.nn.functional as F
import utils.common_utils as common_utils
from data_utils import split_batch
import glob

# plt.interactive(True)

CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_EXTENSION = 'pth.tar'


class Solver(object):

    def __init__(self,
                 model,
                 exp_name,
                 device,
                 num_class,
                 optim=torch.optim.Adam,
                 optim_args={},
                 loss_func=losses.CombinedLoss(),
                 model_name='Segmentor',
                 labels=None,
                 num_epochs=10,
                 log_nth=5,
                 lr_scheduler_step_size=5,
                 lr_scheduler_gamma=0.5,
                 use_last_checkpoint=True,
                 exp_dir='experiments',
                 log_dir='logs'):

        self.device = device
        self.model = model

        self.model_name = model_name
        self.labels = labels
        self.num_epochs = num_epochs
        if torch.cuda.is_available():
            self.loss_func = loss_func.cuda(device)
        else:
            self.loss_func = loss_func
        self.optim = optim(model.parameters(), **optim_args)
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=lr_scheduler_step_size,
                                             gamma=lr_scheduler_gamma)

        exp_dir_path = os.path.join(exp_dir, exp_name)
        self.exp_dir_path = exp_dir_path
        common_utils.create_if_not(exp_dir_path)
        common_utils.create_if_not(os.path.join(exp_dir_path, CHECKPOINT_DIR))

        self.log_nth = log_nth
        self.logWriter = LogWriter(num_class, log_dir, exp_name, use_last_checkpoint, labels)

        self.use_last_checkpoint = use_last_checkpoint
        self.start_epoch = 1
        self.start_iteration = 1

        self.best_ds_mean = 0
        self.best_ds_mean_epoch = 0

        if use_last_checkpoint:
            self.load_checkpoint()

    # TODO:Need to correct the CM and dice score calculation.
    def train(self, train_loader, test_loader):
        """
        Train a given model with the provided data.

        Inputs:
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        """
        model, optim, scheduler = self.model, self.optim, self.scheduler

        data_loader = {
            'train': train_loader,
            'val': test_loader
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model.cuda(self.device)

        print('START TRAINING. : model name = %s, device = %s' % (
            self.model_name, torch.cuda.get_device_name(self.device)))
        current_iteration = self.start_iteration

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            print("\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))
            for phase in ['train', 'val']:
                print("<<<= Phase: %s =>>>" % phase)
                loss_arr = []
                y_list = []
                out_list = []

                if phase == 'train':
                    model.train()
                    scheduler.step()
                else:
                    model.eval()
                for i_batch, sampled_batch in enumerate(data_loader[phase]):
                    X = sampled_batch[0].type(torch.FloatTensor).cuda(
                        self.device, non_blocking=True)
                    y = sampled_batch[1].type(torch.LongTensor).cuda(
                        self.device, non_blocking=True)
                    w = sampled_batch[2].type(torch.FloatTensor).cuda(
                        self.device, non_blocking=True)

                    e1, e2, e3, bn, d3, d2, d1, output = model(X)
                    # TODO: add weights
                    # loss = self.loss_func(output, y2)
                    loss = self.loss_func(output, y, w)
                    if phase == 'train':
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        if i_batch % self.log_nth == 0:
                            self.logWriter.loss_per_iter(loss.item(), i_batch, current_iteration)
                        current_iteration += 1

                    loss_arr.append(loss.item())

                    # batch_output = output > 0.5
                    _, batch_output = torch.max(F.softmax(output, dim=1), dim=1)

                    out_list.append(batch_output.cpu())
                    y_list.append(y.cpu())
                    del X, y, w, output, batch_output, loss, e1, e2, e3, bn, d3, d2, d1
                    torch.cuda.empty_cache()
                    if phase == 'val':
                        if i_batch != len(data_loader[phase]) - 1:
                            print("#", end='', flush=True)
                        else:
                            print("100%", flush=True)

                with torch.no_grad():
                    y_arr = torch.cat(y_list)
                    out_arr = torch.cat(out_list)

                    self.logWriter.loss_per_epoch(loss_arr, phase, epoch)
                    index = np.random.choice(len(out_arr), 3, replace=False)
                    self.logWriter.image_per_epoch_segmentor(out_arr[index], y_arr[index], phase, epoch)
                    ds_mean = self.logWriter.dice_score_per_epoch_segmentor(phase, out_arr, y_arr, epoch)
                    if phase == 'val':
                        if ds_mean > self.best_ds_mean:
                            self.best_ds_mean = ds_mean
                            self.best_ds_mean_epoch = epoch

            print("==== Epoch [" + str(epoch) + " / " + str(self.num_epochs) + "] DONE ====")
            self.save_checkpoint({
                'epoch': epoch + 1,
                'start_iteration': current_iteration + 1,
                'arch': self.model_name,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                            'checkpoint_epoch_' + str(epoch) + '.' + CHECKPOINT_EXTENSION))

        print('FINISH.')
        self.logWriter.close()

    def save_best_model(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        self.load_checkpoint(self.best_ds_mean_epoch)

        torch.save(self.model, path)

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def load_checkpoint(self, epoch=None):
        if epoch is not None:
            checkpoint_path = os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                                           'checkpoint_epoch_' + str(epoch) + '.' + CHECKPOINT_EXTENSION)
            self._load_checkpoint_file(checkpoint_path)
        else:
            all_files_path = os.path.join(self.exp_dir_path, CHECKPOINT_DIR, '*.' + CHECKPOINT_EXTENSION)
            list_of_files = glob.glob(all_files_path)
            if len(list_of_files) > 0:
                checkpoint_path = max(list_of_files, key=os.path.getctime)
                self._load_checkpoint_file(checkpoint_path)
            else:
                self.logWriter.log(
                    "=> no checkpoint found at '{}' folder".format(os.path.join(self.exp_dir_path, CHECKPOINT_DIR)))

    def _load_checkpoint_file(self, file_path):
        self.logWriter.log("=> loading checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path)
        self.start_epoch = checkpoint['epoch']
        self.start_iteration = checkpoint['start_iteration']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer'])

        for state in self.optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.logWriter.log("=> loaded checkpoint '{}' (epoch {})".format(file_path, checkpoint['epoch']))
