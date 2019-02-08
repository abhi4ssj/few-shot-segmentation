import os
import numpy as np
import torch
from nn_additional_losses import losses
from torch.optim import lr_scheduler
import torch.nn as nn
from utils.log_utils import LogWriter
import utils.common_utils as common_utils
from utils.data_utils import split_batch
import glob
import torch.nn.functional as F

# plt.interactive(True)

CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_EXTENSION = 'pth.tar'


class Solver(object):

    def __init__(self,
                 model,
                 exp_name,
                 device,
                 num_class,
                 optim=torch.optim.SGD,
                 optim_args={},
                 loss_func=nn.BCELoss(),
                 model_name='OneShotSegmentor',
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

        # self.optim = optim(model.parameters(), **optim_args)

        self.optim_c = optim(
            [{'params': model.conditioner.parameters(), 'lr': 1e-4, 'momentum': 0.99, 'weight_decay': 0.0001}
             ], **optim_args)

        self.optim_s = optim(
            [{'params': model.segmentor.parameters(), 'lr': 1e-4, 'momentum': 0.99, 'weight_decay': 0.0001}
             ], **optim_args)

        # self.scheduler = lr_scheduler.StepLR(self.optim, step_size=5,
        #                                        gamma=0.1)
        self.scheduler_s = lr_scheduler.StepLR(self.optim_s, step_size=10,
                                               gamma=0.1)
        self.scheduler_c = lr_scheduler.StepLR(self.optim_c, step_size=10,
                                               gamma=0.001)

        exp_dir_path = os.path.join(exp_dir, exp_name)
        common_utils.create_if_not(exp_dir_path)
        common_utils.create_if_not(os.path.join(exp_dir_path, CHECKPOINT_DIR))
        self.exp_dir_path = exp_dir_path

        self.log_nth = log_nth
        self.logWriter = LogWriter(num_class, log_dir, exp_name, use_last_checkpoint, labels)

        self.use_last_checkpoint = use_last_checkpoint
        self.start_epoch = 1
        self.start_iteration = 1

        self.best_ds_mean = 0
        self.best_ds_mean_epoch = 0

        if use_last_checkpoint:
            self.load_checkpoint()

    def train(self, train_loader, test_loader):
        """
        Train a given model with the provided data.

        Inputs:
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        """
        model, optim_c, optim_s, scheduler_c, scheduler_s = self.model, self.optim_c, self.optim_s, self.scheduler_c, self.scheduler_s

        data_loader = {
            'train': train_loader,
            'val': test_loader
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model.cuda(self.device)

        self.logWriter.log('START TRAINING. : model name = %s, device = %s' % (
            self.model_name, torch.cuda.get_device_name(self.device)))
        current_iteration = self.start_iteration
        warm_up_epoch = 15
        val_old = 0
        change_model = False
        current_model = 'seg'
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.logWriter.log('train', "\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))
            if epoch > warm_up_epoch:
                if current_model == 'seg':
                    self.logWriter.log("Optimizing Segmentor")
                    optim = optim_s
                elif current_model == 'con':
                    optim = optim_c
                    self.logWriter.log("Optimizing Conditioner")

            for phase in ['train', 'val']:
                self.logWriter.log("<<<= Phase: %s =>>>" % phase)
                loss_arr = []
                input_img_list = []
                y_list = []
                out_list = []
                condition_input_img_list = []
                condition_y_list = []

                if phase == 'train':
                    model.train()
                    scheduler_c.step()
                    scheduler_s.step()
                else:
                    model.eval()
                for i_batch, sampled_batch in enumerate(data_loader[phase]):
                    X = sampled_batch[0].type(torch.FloatTensor)
                    y = sampled_batch[1].type(torch.LongTensor)
                    w = sampled_batch[2].type(torch.FloatTensor)

                    query_label = data_loader[phase].batch_sampler.query_label

                    input1, input2, y1, y2 = split_batch(X, y, int(query_label))

                    condition_input = torch.cat((input1, y1.unsqueeze(1)), dim=1)

                    query_input = input2
                    y1 = y1.type(torch.LongTensor)
                    # TODO: Only for shaban baseline
                    y2 = y2.type(torch.FloatTensor)

                    if model.is_cuda:
                        condition_input, query_input, y2, y1 = condition_input.cuda(self.device,
                                                                                    non_blocking=True), query_input.cuda(
                            self.device,
                            non_blocking=True), y2.cuda(
                            self.device, non_blocking=True), y1.cuda(
                            self.device, non_blocking=True)

                    weights = model.conditioner(condition_input)

                    output = model.segmentor(query_input, weights)
                    # TODO: add weights
                    # loss_weights = (1, 0) if epoch < 5 else (0.5, 0.5)
                    # loss = self.loss_func(F.softmax(output, dim=1), y2)
                    loss = self.loss_func(torch.sigmoid(output), y2)
                    optim_s.zero_grad()
                    optim_c.zero_grad()
                    loss.backward()
                    if phase == 'train':
                        if epoch <= warm_up_epoch:
                            optim_s.step()
                            optim_c.step()
                        elif epoch > warm_up_epoch and change_model:
                            optim.step()

                        # # TODO: value needs to be optimized, Gradient Clipping (Optional)
                        # if epoch > 1:
                        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)

                        if i_batch % self.log_nth == 0:
                            self.logWriter.loss_per_iter(loss.item(), i_batch, current_iteration)
                        current_iteration += 1

                    loss_arr.append(loss.item())

                    batch_output = output.squeeze() > 0.5

                    # _, batch_output = torch.max(F.softmax(output, dim=1), dim=1)
                    batch_output.cpu()
                    batch_output.type(torch.FloatTensor)
                    out_list.append(batch_output)
                    input_img_list.append(input2.cpu())
                    y_list.append(y2.cpu())
                    condition_input_img_list.append(input1.cpu())
                    condition_y_list.append(y1)

                    del X, y, w, output, batch_output, loss, input1, input2, y2
                    torch.cuda.empty_cache()
                    if phase == 'val':
                        if i_batch != len(data_loader[phase]) - 1:
                            print("#", end='', flush=True)
                        else:
                            print("100%", flush=True)
                if phase == 'train':
                    self.logWriter.log('saving checkpoint ....')
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'start_iteration': current_iteration + 1,
                        'arch': self.model_name,
                        'state_dict': model.state_dict(),
                        'optimizer_c': optim_c.state_dict(),
                        'scheduler_c': scheduler_c.state_dict(),
                        'optimizer_s': optim_s.state_dict(),
                        'best_ds_mean_epoch': self.best_ds_mean_epoch,
                        'scheduler_s': scheduler_s.state_dict()
                    }, os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                                    'checkpoint_epoch_' + str(epoch) + '.' + CHECKPOINT_EXTENSION))

                with torch.no_grad():
                    input_img_arr = torch.cat(input_img_list)
                    y_arr = torch.cat(y_list)
                    out_arr = torch.cat(out_list)
                    condition_input_img_arr = torch.cat(condition_input_img_list)
                    condition_y_arr = torch.cat(condition_y_list)

                    current_loss = self.logWriter.loss_per_epoch(loss_arr, phase, epoch)
                    if phase == 'val':
                        if epoch > warm_up_epoch:
                            self.logWriter.log("Diff : " + str(current_loss - val_old))
                            change_model = (current_loss - val_old) > 0.001

                        if change_model and current_model == 'seg':
                            self.logWriter.log("Setting to con")
                            current_model = 'con'
                        elif change_model and current_model == 'con':
                            self.logWriter.log("Setting to seg")
                            current_model = 'seg'
                        val_old = current_loss
                    index = np.random.choice(len(out_arr), 3, replace=False)
                    self.logWriter.image_per_epoch(out_arr[index], y_arr[index], phase, epoch, additional_image=(
                        input_img_arr[index], condition_input_img_arr[index], condition_y_arr[index]))
                    ds_mean = self.logWriter.dice_score_per_epoch(phase, out_arr, y_arr, epoch)
                    if phase == 'val':
                        if ds_mean > self.best_ds_mean:
                            self.best_ds_mean = ds_mean
                            self.best_ds_mean_epoch = epoch

                    self.logWriter.log("==== Epoch [" + str(epoch) + " / " + str(self.num_epochs) + "] DONE ====")
                self.logWriter.log('FINISH.')
        self.logWriter.close()

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def save_best_model(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        print("Best Epoch... " + str(self.best_ds_mean_epoch))
        self.load_checkpoint(self.best_ds_mean_epoch)

        torch.save(self.model, path)

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

    # def _load_checkpoint_file(self, file_path):
    #     self.logWriter.log("=> loading checkpoint '{}'".format(file_path))
    #     checkpoint = torch.load(file_path)
    #     self.start_epoch = checkpoint['epoch']
    #     self.start_iteration = checkpoint['start_iteration']
    #     self.model.load_state_dict(checkpoint['state_dict'])
    #     self.optim.load_state_dict(checkpoint['optimizer'])
    #
    #     for state in self.optim.state.values():
    #         for k, v in state.items():
    #             if torch.is_tensor(v):
    #                 state[k] = v.to(self.device)
    #
    #     self.scheduler.load_state_dict(checkpoint['scheduler'])
    #     self.logWriter.log("=> loaded checkpoint '{}' (epoch {})".format(file_path, checkpoint['epoch']))

    def _load_checkpoint_file(self, file_path):
        self.logWriter.log("=> loading checkpoint '{}'".format(file_path))
        # checkpoint = torch.load(file_path)
        # checkpoint_path = os.path.join(self.exp_dir_path, CHECKPOINT_DIR, '*.' + CHECKPOINT_EXTENSION)
        # list_of_files = glob.glob(checkpoint_path)
        # if len(list_of_files) > 0:
        #     latest_file = max(list_of_files, key=os.path.getctime)
        #     self.logWriter.log("=> loading checkpoint '{}'".format(latest_file))
        checkpoint = torch.load(file_path)
        self.start_epoch = checkpoint['epoch']
        self.start_iteration = checkpoint['start_iteration']
        self.best_ds_mean_epoch = checkpoint['best_ds_mean_epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optim_c.load_state_dict(checkpoint['optimizer_c'])
        self.optim_s.load_state_dict(checkpoint['optimizer_s'])

        for state in self.optim_c.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        for state in self.optim_s.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        self.scheduler_c.load_state_dict(checkpoint['scheduler_c'])
        self.scheduler_s.load_state_dict(checkpoint['scheduler_s'])
        self.logWriter.log("=> loaded checkpoint '{}' (epoch {})".format(file_path, checkpoint['epoch']))
        # else:
        #     self.logWriter.log(
        #         "=> no checkpoint found at '{}' folder".format(os.path.join(self.exp_dir_path, CHECKPOINT_DIR)))
