import argparse
import os

import torch
import torch.nn as nn

import utils.evaluator as eu
import few_shot_segmentor_model1 as fs1
import few_shot_segmentor_model2 as fs2
import few_shot_segmentor_model3 as fs3
import few_shot_segmentor_model4 as fs4
import few_shot_segmentor_model5 as fs5
import few_shot_segmentor_model6 as fs6
import few_shot_segmentor_model7 as fs7
import few_shot_segmentor_model8 as fs8
import few_shot_segmentor_model9 as fs9
import few_shot_segmentor_model10 as fs10
import few_shot_segmentor_model11 as fs11
from settings import Settings
# from solver_oneshot_singleOpti import Solver
from solver_oneshot_multiOpti_auto import Solver
from utils.data_utils import get_imdb_dataset
from utils.log_utils import LogWriter
from utils.shot_batch_sampler import OneShotBatchSampler
from utils.shot_batch_sampler import get_lab_list

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
    train_data, test_data = load_data(data_params)

    folds = ['fold1']
    model_prefix = 'model6_wholebody_condch16_e4Skip_inter_e3e4bnd4d3d2_allSseSeg_DiceLoss_'
    for fold in folds:
        final_model_path = os.path.join(common_params['save_model_dir'], model_prefix + fold + '.pth.tar')

        train_params['exp_name'] = model_prefix + fold

        train_sampler = OneShotBatchSampler(train_data.y, 'train', fold, train_params['train_batch_size'],
                                            iteration=train_params['iterations'])
        test_sampler = OneShotBatchSampler(test_data.y, 'val', fold, train_params['val_batch_size'],
                                           iteration=train_params['test_iterations'])

        train_loader = torch.utils.data.DataLoader(train_data, batch_sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(test_data, batch_sampler=test_sampler)

        # segmentor_pretrained = torch.load(train_params['pre_trained_path'])
        # conditioner_pretrained = torch.load(train_params['pre_trained_path'])
        # segmentor_pretrained.classifier = Identity()
        # segmentor_pretrained.sigmoid = Identity()

        # for param in segmentor_pretrained.parameters():
        #     param.requires_grad = False

        few_shot_model = fs6.FewShotSegmentorDoubleSDnet(net_params)
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
    folds = ['fold1']

    eval_model_path1 = "saved_models/model6_wholebody_condch16_e4Skip_inter_e3e4bnd4d3d2_allSseSeg_DiceLoss_fold1.pth.tar"
    eval_model_path2 = "saved_models/model6_coronal_wholebody_condch16_e4Skip_inter_e3e4bnd4d3d2_allSseSeg_DiceLoss_fold1.pth.tar"
    # eval_model_path3 = "saved_models/model6_sagittal_fold1.pth.tar"

    orientaion1 = 'AXI'
    # orientaion2 = 'COR'

    for fold in folds:
        # eval_model_path = os.path.join('saved_models', model_name + '_' + fold + '.pth.tar')
        query_labels = get_lab_list('val', fold)
        num_classes = len(fold)

        #avg_dice_score = eu.evaluate_dice_score(eval_model_path1,
        #                                       num_classes,
        #                                        query_labels,
        #                                        data_dir,
        #                                        query_txt_file,
        #                                        support_txt_file,
        #                                        remap_config,
        #                                        orientaion1,
        #                                        prediction_path,
        #                                        device,
        #                                        logWriter, fold=fold)

        #avg_dice_score = eu.evaluate_dice_score_3view(eval_model_path1,
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
        avg_dice_score = eu.evaluate_dice_score_2view(eval_model_path1,
                                                      eval_model_path2,
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

        logWriter.log(avg_dice_score)
    logWriter.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train and eval')
    args = parser.parse_args()

    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    if args.mode == 'train':
        train(train_params, common_params, data_params, net_params)
    elif args.mode == 'eval':
        evaluate(eval_params, net_params, data_params, common_params, train_params)
    else:
        raise ValueError('Invalid value for mode. only support values are train and eval')
