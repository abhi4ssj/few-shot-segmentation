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
                 loss_func=losses.DiceLossBinary(),
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

        self.optim = optim(
            [{'params': model.conditioner.parameters(), 'lr': 1e-3, 'momentum': 0.99, 'weight_decay': 0.001},
             {'params': model.segmentor.parameters(), 'lr': 1e-3, 'momentum': 0.99, 'weight_decay': 0.001},
             ], **optim_args)
        # self.optim = optim(model.parameters(), **optim_args)
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=lr_scheduler_step_size,
                                             gamma=lr_scheduler_gamma)

        exp_dir_path = os.path.join(exp_dir, exp_name)
        common_utils.create_if_not(exp_dir_path)
        common_utils.create_if_not(os.path.join(exp_dir_path, CHECKPOINT_DIR))
        self.exp_dir_path = exp_dir_path

        self.log_nth = log_nth
        self.logWriter = LogWriter(num_class, log_dir, exp_name, use_last_checkpoint, labels)

        self.use_last_checkpoint = use_last_checkpoint
        self.start_epoch = 1
        self.start_iteration = 1

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

                    if model.is_cuda:
                        condition_input, input2, y2 = condition_input.cuda(self.device, non_blocking=True), input2.cuda(
                            self.device,
                            non_blocking=True), y2.cuda(
                            self.device, non_blocking=True)

                    output = model(condition_input, input2)
                    # TODO: add weights
                    loss = self.loss_func(output, y2)
                    if phase == 'train':
                        optim.zero_grad()
                        loss.backward()
                        # # TODO: value needs to be optimized, Gradient Clipping (Optional)
                        # if epoch > 1:
                        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
                        optim.step()
                        if i_batch % self.log_nth == 0:
                            self.logWriter.loss_per_iter(loss.item(), i_batch, current_iteration)
                        current_iteration += 1

                    loss_arr.append(loss.item())

                    batch_output = output > 0.5

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

                with torch.no_grad():
                    input_img_arr = torch.cat(input_img_list)
                    y_arr = torch.cat(y_list)
                    out_arr = torch.cat(out_list)
                    condition_input_img_arr = torch.cat(condition_input_img_list)
                    condition_y_arr = torch.cat(condition_y_list)

                    self.logWriter.loss_per_epoch(loss_arr, phase, epoch)
                    index = np.random.choice(len(out_arr), 3, replace=False)
                    self.logWriter.image_per_epoch(out_arr[index], y_arr[index], phase, epoch, additional_image=(
                        input_img_arr[index], condition_input_img_arr[index], condition_y_arr[index]))
                    self.logWriter.dice_score_per_epoch(phase, out_arr, y_arr, epoch)

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

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.exp_dir_path, CHECKPOINT_DIR, '*.' + CHECKPOINT_EXTENSION)
        list_of_files = glob.glob(checkpoint_path)
        if len(list_of_files) > 0:
            latest_file = max(list_of_files, key=os.path.getctime)
            print("=> loading checkpoint '{}'".format(latest_file))
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
            print("=> loaded checkpoint '{}' (epoch {})".format(latest_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}' folder".format(os.path.join(self.exp_dir_path, CHECKPOINT_DIR)))
