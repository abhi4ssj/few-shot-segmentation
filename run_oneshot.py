import argparse
import os

import torch
import torch.nn as nn

import utils.evaluator as eu
from few_shot_segmentor import Segmentor
from few_shot_segmentor_baseline import FewShotSegmentorBaseLine
from settings import Settings
from solver_oneshot import Solver
from utils.data_utils import get_imdb_dataset
from utils.log_utils import LogWriter
from utils.shot_batch_sampler import OneShotBatchSampler

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

    train_sampler = OneShotBatchSampler(train_data.y, 'train', train_params['train_batch_size'],
                                        iteration=train_params['iterations'])
    test_sampler = OneShotBatchSampler(test_data.y, 'val', train_params['val_batch_size'],
                                       iteration=train_params['test_iterations'])

    train_loader = torch.utils.data.DataLoader(train_data, batch_sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(test_data, batch_sampler=test_sampler)

    # segmentor_pretrained = torch.load(train_params['pre_trained_path'])
    #
    # segmentor_pretrained.classifier = Identity()
    # segmentor_pretrained.sigmoid = Identity()

    # for param in segmentor_pretrained.parameters():
    #     param.requires_grad = False

    few_shot_model = FewShotSegmentorBaseLine(net_params)
    # few_shot_model.segmentor = segmentor_pretrained

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
    final_model_path = os.path.join(common_params['save_model_dir'], train_params['final_model_file'])
    few_shot_model.save(final_model_path)
    print("final model saved @ " + str(final_model_path))


def evaluate(eval_params, net_params, data_params, common_params, train_params):
    eval_model_path = eval_params['eval_model_path']
    num_classes = net_params['num_class']
    labels = data_params['labels']
    data_dir = eval_params['data_dir']
    label_dir = eval_params['label_dir']
    volumes_txt_file = eval_params['volumes_txt_file']
    remap_config = eval_params['remap_config']
    device = common_params['device']
    log_dir = common_params['log_dir']
    exp_dir = common_params['exp_dir']
    exp_name = train_params['exp_name']
    save_predictions_dir = eval_params['save_predictions_dir']
    prediction_path = os.path.join(exp_dir, exp_name, save_predictions_dir)
    orientation = eval_params['orientation']

    logWriter = LogWriter(num_classes, log_dir, exp_name, labels=labels)

    avg_dice_score, class_dist = eu.evaluate_dice_score(eval_model_path,
                                                        num_classes,
                                                        data_dir,
                                                        label_dir,
                                                        volumes_txt_file,
                                                        remap_config,
                                                        orientation,
                                                        prediction_path,
                                                        device,
                                                        logWriter)
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
