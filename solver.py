import os

import numpy as np
import torch
from nn_additional_losses import losses as additional_losses
from torch.optim import lr_scheduler

from utils.log_utils import LogWriter
import utils.common_utils as common_utils

CHECKPOINT_FILE_NAME = 'checkpoint.pth.tar'


class Solver(object):

    def __init__(self,
                 model,
                 exp_name,
                 device,
                 num_class,
                 optim=torch.optim.Adam,
                 optim_args={},
                 loss_func=additional_losses.CombinedLoss(),
                 model_name='quicknat',
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
        common_utils.create_if_not(exp_dir_path)
        self.exp_dir_path = exp_dir_path

        self.log_nth = log_nth
        self.logWriter = LogWriter(num_class, log_dir, exp_name, use_last_checkpoint, labels)

        self.use_last_checkpoint = use_last_checkpoint
        self.checkpoint_path = os.path.join(exp_dir_path, CHECKPOINT_FILE_NAME)
        self.start_epoch = 1
        self.start_iteration = 1
        if use_last_checkpoint:
            self.load_checkpoint()

    # TODO:Need to correct the CM and dice score calculation.
    def train(self, train_loader, val_loader):
        """
        Train a given model with the provided data.

        Inputs:
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        """
        model, optim, scheduler = self.model, self.optim, self.scheduler
        dataloaders = {
            'train': train_loader,
            'val': val_loader
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
                out_list = []
                y_list = []
                if phase == 'train':
                    model.train()
                    scheduler.step()
                else:
                    model.eval()
                for i_batch, sample_batched in enumerate(dataloaders[phase]):
                    X = sample_batched[0].type(torch.FloatTensor)
                    y = sample_batched[1].type(torch.LongTensor)
                    w = sample_batched[2].type(torch.FloatTensor)

                    if model.is_cuda:
                        X, y, w = X.cuda(self.device, non_blocking=True), y.cuda(self.device,
                                                                                 non_blocking=True), w.cuda(self.device,
                                                                                                            non_blocking=True)

                    output = model(X)
                    loss = self.loss_func(output, y, w)
                    if phase == 'train':
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        if i_batch % self.log_nth == 0:
                            self.logWriter.loss_per_iter(loss.item(), i_batch, current_iteration)
                        current_iteration += 1

                    loss_arr.append(loss.item())

                    _, batch_output = torch.max(output, dim=1)
                    out_list.append(batch_output.cpu())
                    y_list.append(y.cpu())

                    del X, y, w, output, batch_output, loss
                    torch.cuda.empty_cache()
                    if phase == 'val':
                        if i_batch != len(dataloaders[phase]) - 1:
                            print("#", end='', flush=True)
                        else:
                            print("100%", flush=True)

                with torch.no_grad():
                    out_arr, y_arr = torch.cat(out_list), torch.cat(y_list)
                    self.logWriter.loss_per_epoch(loss_arr, phase, epoch)
                    index = np.random.choice(len(dataloaders[phase].dataset.X), 3, replace=False)
                    self.logWriter.image_per_epoch(model.predict(dataloaders[phase].dataset.X[index], self.device),
                                                   dataloaders[phase].dataset.y[index], phase, epoch)
                    self.logWriter.cm_per_epoch(phase, out_arr, y_arr, epoch)
                    self.logWriter.dice_score_per_epoch(phase, out_arr, y_arr, epoch)

            print("==== Epoch [" + str(epoch) + " / " + str(self.num_epochs) + "] DONE ====")
            self.save_checkpoint({
                'epoch': epoch + 1,
                'start_iteration': current_iteration + 1,
                'arch': self.model_name,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict()
            }, self.checkpoint_path)

        print('FINISH.')
        self.logWriter.close()

    def save_checkpoint(self, state, filename=CHECKPOINT_FILE_NAME):
        torch.save(state, filename)

    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            print("=> loading checkpoint '{}'".format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch']
            self.start_iteration = checkpoint['start_iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])

            for state in self.optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(self.checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.checkpoint_path))
