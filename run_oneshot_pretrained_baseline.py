import argparse
import os
import numpy as np
import torch
import torch.nn as nn

import utils.evaluator_finetuning as eu
import few_shot_segmentor_pretrained as fs
from settings import Settings
from solver_pretrained import Solver
from utils.evaluator import binarize_label
from utils.data_utils import ImdbData

import utils.data_utils as du
from utils.data_utils import get_imdb_dataset
from utils.log_utils import LogWriter
from utils.shot_batch_sampler import get_lab_list
from nn_common_modules import modules as sm

torch.set_default_tensor_type('torch.FloatTensor')


def load_data(data_params):
    print("Loading dataset")
    train_data, test_data = get_imdb_dataset(data_params)
    print("Train size: %i" % len(train_data))
    print("Test size: %i" % len(test_data))
    return train_data, test_data


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def train(train_params, common_params, data_params, net_params):
    query_label = 8
    Num_support = 10

    # train_data, test_data = load_data(data_params)

    support_volume, support_labelmap, _, _ = du.load_and_preprocess(
        "/home/deeplearning/Abhijit/nas_drive/Abhijit/WholeBody/CT_ce/Data/Visceral/10000132_1_CTce_ThAb.mat",
        orientation='AXI',
        remap_config="WholeBody")
    support_volume = support_volume if len(support_volume.shape) == 4 else support_volume[:, np.newaxis, :,
                                                                           :]
    support_volume, support_labelmap = torch.tensor(support_volume).type(torch.FloatTensor), \
                                       torch.tensor(support_labelmap).type(torch.LongTensor)

    support_labelmap = (support_labelmap == query_label).type(torch.FloatTensor)
    batch, _, _ = support_labelmap.size()
    slice_with_class = torch.sum(support_labelmap.view(batch, -1), dim=1) > 10
    support_labelmap = support_labelmap[slice_with_class]
    support_volume = support_volume[slice_with_class]

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support + 1)).astype(int)
    support_slice_indexes += (len(support_volume) // Num_support) // 2
    support_slice_indexes = support_slice_indexes[:-1]
    support_volume = support_volume[support_slice_indexes]
    support_labelmap = support_labelmap[support_slice_indexes]

    train_data = ImdbData(support_volume.numpy(), support_labelmap.numpy(), np.ones_like(support_labelmap.numpy()))

    # Removing unused classes
    # train_data.y[train_data.y == 3] = 0
    # test_data.y[test_data.y == 3] = 0
    #
    # train_data.y[train_data.y == 4] = 0
    # test_data.y[test_data.y == 4] = 0
    #
    # train_data.y[train_data.y == 5] = 0
    # test_data.y[test_data.y == 5] = 0
    #
    # train_data.y[train_data.y == 6] = 3
    # test_data.y[test_data.y == 6] = 3
    #
    # train_data.y[train_data.y == 7] = 4
    # test_data.y[test_data.y == 7] = 4
    #
    # train_data.y[train_data.y == 8] = 0
    # test_data.y[test_data.y == 8] = 0
    #
    # train_data.y[train_data.y == 9] = 0
    # test_data.y[test_data.y == 9] = 0

    # batch_size = len(train_data.y)
    # non_black_slices = np.sum(train_data.y.reshape(batch_size, -1), axis=1) > 10
    # train_data.X = train_data.X[non_black_slices]
    # train_data.y = train_data.y[non_black_slices]

    # batch_size = len(test_data.y)
    # non_black_slices = np.sum(test_data.y.reshape(batch_size, -1), axis=1) > 10
    # test_data.X = test_data.X[non_black_slices]
    # test_data.y = test_data.y[non_black_slices]

    model_prefix = 'finetuned_segmentor_'
    folds = ['fold4']
    for fold in folds:
        final_model_path = os.path.join(common_params['save_model_dir'], model_prefix + fold + '.pth.tar')

        train_params['exp_name'] = model_prefix + fold

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_params['train_batch_size'],
                                                   shuffle=True,
                                                   num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(train_data, batch_size=train_params['val_batch_size'], shuffle=False,
                                                 num_workers=4, pin_memory=True)

        # conditioner_pretrained = torch.load(train_params['pre_trained_path'])
        # segmentor_pretrained.classifier = Identity()
        # segmentor_pretrained.sigmoid = Identity()

        # for param in segmentor_pretrained.parameters():
        #     param.requires_grad = False

        # few_shot_model = fs.SDnetSegmentor(net_params)
        few_shot_model = torch.load(train_params['pre_trained_path'])
        for param in few_shot_model.parameters():
            param.requires_grad = False
        net_params['num_channels'] = 64
        few_shot_model.classifier = sm.ClassifierBlock(net_params)
        for param in few_shot_model.classifier.parameters():
            param.requires_grad = True

        # few_shot_model = segmentor_pretrained
        # few_shot_model.conditioner = conditioner_pretrained

        solver = Solver(few_shot_model,
                        device=common_params['device'],
                        num_class=net_params['num_class'],
                        optim_args={"lr": train_params['learning_rate'],
                                    # "betas": train_params['optim_betas'],
                                    # "eps": train_params['optim_eps'],
                                    "weight_decay": train_params['optim_weight_decay'],
                                    "momentum": train_params['momentum']},
                        model_name=common_params['model_name'],
                        exp_name=train_params['exp_name'],
                        labels=data_params['labels'],
                        log_nth=train_params['log_nth'],
                        num_epochs=train_params['num_epochs'],
                        lr_scheduler_step_size=train_params['lr_scheduler_step_size'],
                        lr_scheduler_gamma=train_params['lr_scheduler_gamma'],
                        use_last_checkpoint=train_params['use_last_checkpoint'],
                        log_dir=common_params['log_dir'],
                        exp_dir=common_params['exp_dir'])

        solver.train(train_loader, val_loader)

        # few_shot_model.save(final_model_path)
        # final_model_path = os.path.join(common_params['save_model_dir'], )
        solver.save_best_model(final_model_path)
        print("final model saved @ " + str(final_model_path))


lab_list_fold = {"fold1": {"train": [2, 6, 7, 8, 9], "val": [1]},
                 "fold2": {"train": [1, 6, 7, 8, 9], "val": [2]},
                 "fold3": {"train": [1, 2, 8, 9], "val": [6]},
                 "fold4": {"train": [1, 2, 6, 7], "val": [8]}}


# For brain
# lab_list_fold = {"fold1": {"train": [1, 2, 3, 5, 6], "val": [4]},
#                  "fold2": {"train": [1, 3, 4, 5, 6], "val": [2]},
#                  "fold3": {"train": [1, 2, 3, 4, 6], "val": [5]},
#                  "fold4": {"train": [1, 2, 3, 4, 5], "val": [6]}}

# lab_list_fold = {"fold1": {"train": [2, 4, 6, 8], "val": [1]},
#                  "fold2": {"train": [1, 4, 6, 8], "val": [2]},
#                  "fold3": {"train": [1, 2, 6, 8], "val": [4]},
#                  "fold4": {"train": [1, 2, 4, 8], "val": [6]},
#                  "fold5": {"train": [1, 2, 4, 6], "val": [8]}}


def _get_lab_list(phase, fold):
    return lab_list_fold[fold][phase]


def evaluate(eval_params, net_params, data_params, common_params, train_params):
    eval_model_path = eval_params['eval_model_path']
    num_classes = net_params['num_class']
    labels = data_params['labels']
    data_dir = eval_params['data_dir']
    label_dir = eval_params['label_dir']
    query_txt_file = eval_params['query_txt_file']
    support_txt_file = eval_params['support_txt_file']
    remap_config = eval_params['remap_config']
    device = common_params['device']
    log_dir = common_params['log_dir']
    exp_dir = common_params['exp_dir']
    exp_name = train_params['exp_name']
    save_predictions_dir = eval_params['save_predictions_dir']
    prediction_path = os.path.join(exp_dir, exp_name, save_predictions_dir)
    orientation = eval_params['orientation']

    logWriter = LogWriter(num_classes, log_dir, exp_name, labels=labels)

    # model_name = 'model6_Dice_L2_loss_target_fold1.pth.tar'
    folds = ['fold4']

    # eval_model_path1 = "saved_models/sne_position_all_type_spatial_fold1.pth.tar"
    eval_model_path1 = "finetuned_segmentor"
    eval_model_path2 = "saved_models/model6_coronal_wholebody_condch16_e4Skip_inter_e3e4bnd4d3_ch_e1e2d1d2_noSseSeg_DiceLoss_lowrate_fold2.pth.tar"
    # eval_model_path3 = "saved_models/model6_sagittal_fold1.pth.tar"

    orientaion1 = 'AXI'
    orientaion2 = 'COR'

    for fold in folds:
        eval_model_path = os.path.join('saved_models', eval_model_path1 + '_' + fold + '.pth.tar')
        query_labels = _get_lab_list('val', fold)
        num_classes = len(fold)

        avg_dice_score = eu.evaluate_dice_score(eval_model_path,
                                                num_classes,
                                                query_labels,
                                                data_dir,
                                                query_txt_file,
                                                support_txt_file,
                                                remap_config,
                                                orientaion1,
                                                prediction_path,
                                                device,
                                                logWriter, fold=fold)

        # avg_dice_score = eu.evaluate_dice_score_3view(eval_model_path1,
        #                                               eval_model_path2,
        #                                               eval_model_path3,
        #                                               num_classes,
        #                                               query_labels,
        #                                               data_dir,
        #                                               query_txt_file,
        #                                               support_txt_file,
        #                                               remap_config,
        #                                               orientaion1,
        #                                               prediction_path,
        #                                               device,
        #                                               logWriter, fold=fold)
        # avg_dice_score = eu.evaluate_dice_score_2view(eval_model_path1,
        #                                               eval_model_path2,
        #                                               num_classes,
        #                                               query_labels,
        #                                               data_dir,
        #                                               query_txt_file,
        #                                               support_txt_file,
        #                                               remap_config,
        #                                               orientaion1,
        #                                               prediction_path,
        #                                               device,
        #                                               logWriter, fold=fold)

        logWriter.log(avg_dice_score)
    logWriter.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train and eval')
    parser.add_argument('--device', '-d', required=False, help='device to run on')
    args = parser.parse_args()

    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    if args.device is not None:
        common_params['device'] = args.device

    if args.mode == 'train':
        train(train_params, common_params, data_params, net_params)
    elif args.mode == 'eval':
        evaluate(eval_params, net_params, data_params, common_params, train_params)
    else:
        raise ValueError('Invalid value for mode. only support values are train and eval')
