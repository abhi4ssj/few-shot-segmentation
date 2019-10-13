import os
import numpy as np
import torch
from nn_additional_losses import losses
from torch.optim import lr_scheduler
import torch.nn as nn
from utils.log_utils import LogWriter
import utils.common_utils as common_utils
from data_utils import split_batch
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
                 loss_func=losses.DiceLoss(),
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

        self.optim = optim([{'params': model.squeeze_conv_bn.parameters()},
                            {'params': model.squeeze_conv_d1.parameters()},
                            {'params': model.squeeze_conv_d2.parameters()},
                            {'params': model.squeeze_conv_d3.parameters()},
                            {'params': model.conditioner.parameters(), 'lr': 1e-2, 'momentum': 0.95, 'weight_decay': 0.001},
                            {'params': model.segmentor.parameters(), 'lr': 1e-2, 'momentum': 0.95,
                             'weight_decay': 0.001}], **optim_args)

        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=2,
                                               gamma=0.1)

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
        model, optim, scheduler = self.model, self.optim, self.scheduler

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
        warm_up_epoch = 5
        val_old = 0
        change_model = False
        current_model = 'seg'
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.logWriter.log('train', "\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))

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
                    scheduler.step()
                else:
                    model.eval()
                for i_batch, sampled_batch in enumerate(data_loader[phase]):
                    X = sampled_batch[0].type(torch.FloatTensor)
                    y = sampled_batch[1].type(torch.LongTensor)
                    w = sampled_batch[2].type(torch.FloatTensor)

                    query_label = data_loader[phase].batch_sampler.query_label

                    input1, input2, y1, y2 = split_batch(X, y, int(query_label))

                    condition_input = torch.mul(input1, y1.unsqueeze(1))
                    query_input = input2

                    if model.is_cuda:
                        condition_input, query_input, y2 = condition_input.cuda(self.device, non_blocking=True), query_input.cuda(
                            self.device,
                            non_blocking=True), y2.cuda(
                            self.device, non_blocking=True)

                    output = model(condition_input, query_input)
                    # TODO: add weights
                    loss = self.loss_func(output, y2)
                    optim.zero_grad()
                    loss.backward()
                    if phase == 'train':
                        optim.step()

                        if i_batch % self.log_nth == 0:
                            self.logWriter.loss_per_iter(loss.item(), i_batch, current_iteration)
                        current_iteration += 1

                    loss_arr.append(loss.item())

                    # batch_output = output > 0.5
                    _, batch_output = torch.max(F.softmax(output, dim=1), dim=1)

                    out_list.append(batch_output.cpu())
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
                        'optimizer': optim.state_dict(),
                        'scheduler': scheduler.state_dict(),
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
                    self.logWriter.dice_score_per_epoch(phase, out_arr, y_arr, epoch)

                    self.logWriter.log("==== Epoch [" + str(epoch) + " / " + str(self.num_epochs) + "] DONE ====")
                self.logWriter.log('FINISH.')
        self.logWriter.close()

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.exp_dir_path, CHECKPOINT_DIR, '*.' + CHECKPOINT_EXTENSION)
        list_of_files = glob.glob(checkpoint_path)
        if len(list_of_files) > 0:
            latest_file = max(list_of_files, key=os.path.getctime)
            self.logWriter.log("=> loading checkpoint '{}'".format(latest_file))
            checkpoint = torch.load(latest_file)
            self.start_epoch = checkpoint['epoch']
            self.start_iteration = checkpoint['start_iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])

            for state in self.optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.logWriter.log("=> loaded checkpoint '{}' (epoch {})".format(latest_file, checkpoint['epoch']))
        else:
            self.logWriter.log(
                "=> no checkpoint found at '{}' folder".format(os.path.join(self.exp_dir_path, CHECKPOINT_DIR)))
