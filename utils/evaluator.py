import os

import nibabel as nib
import numpy as np
import torch

import utils.common_utils as common_utils
import utils.data_utils as du
import shot_batch_sampler as SB


def dice_score_binary(vol_output, ground_truth, no_samples=10, phase='train'):
    ground_truth = ground_truth.type(torch.FloatTensor)
    vol_output = vol_output.type(torch.FloatTensor)
    if phase == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    inter = 2 * torch.sum(torch.mul(ground_truth, vol_output))
    union = torch.sum(ground_truth) + torch.sum(vol_output) + 0.0001

    return torch.div(inter, union)


def dice_confusion_matrix(vol_output, ground_truth, num_classes, no_samples=10, mode='train'):
    dice_cm = torch.zeros(num_classes, num_classes)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        for j in range(num_classes):
            Pred = (vol_output == j).float()
            inter = torch.sum(torch.mul(GT, Pred))
            union = torch.sum(GT) + torch.sum(Pred) + 0.0001
            dice_cm[i, j] = 2 * torch.div(inter, union)
    avg_dice = torch.mean(torch.diagflat(dice_cm))
    return avg_dice, dice_cm


def dice_score_perclass(vol_output, ground_truth, num_classes, no_samples=10, mode='train'):
    dice_perclass = torch.zeros(num_classes)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred))
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
    return dice_perclass


def binarize_label(volume, groud_truth, class_label):
    groud_truth = (groud_truth == class_label).type(torch.FloatTensor)
    condition_input = torch.mul(volume, groud_truth.unsqueeze(1))
    return condition_input


def evaluate_dice_score(model_path,
                        num_classes,
                        query_labels,
                        data_dir,
                        query_txt_file,
                        support_txt_file,
                        remap_config,
                        orientation,
                        prediction_path, device=0, logWriter=None, mode='eval', fold=None):
    print("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")
    print("Loading model => " + model_path)
    batch_size = 10

    with open(query_txt_file) as file_handle:
        volumes_query = file_handle.read().splitlines()

    # with open(support_txt_file) as file_handle:
    #     volumes_support = file_handle.read().splitlines()

    model = torch.load(model_path)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        model.cuda(device)

    model.eval()

    common_utils.create_if_not(prediction_path)

    print("Evaluating now... " + fold)
    query_file_paths = du.load_file_paths(data_dir, data_dir, query_txt_file)
    support_file_paths = du.load_file_paths(data_dir, data_dir, support_txt_file)

    with torch.no_grad():
        all_query_dice_score_list = []
        for query_label in query_labels:
            volume_dice_score_list = []
            for vol_idx, file_path in enumerate(support_file_paths):
                # Loading support
                support_volume, support_labelmap, _, _ = du.load_and_preprocess(file_path,
                                                                                orientation=orientation,
                                                                                remap_config=remap_config)
                support_volume = support_volume if len(support_volume.shape) == 4 else support_volume[:, np.newaxis, :,
                                                                                       :]
                support_volume, support_labelmap = torch.tensor(support_volume).type(torch.FloatTensor), torch.tensor(
                    support_labelmap).type(torch.LongTensor)
                support_volume = binarize_label(support_volume, support_labelmap, query_label)

            for vol_idx, file_path in enumerate(query_file_paths):
                query_volume, query_labelmap, _, _ = du.load_and_preprocess(file_path,
                                                                            orientation=orientation,
                                                                            remap_config=remap_config)

                query_volume = query_volume if len(query_volume.shape) == 4 else query_volume[:, np.newaxis, :, :]
                query_volume, query_labelmap = torch.tensor(query_volume).type(torch.FloatTensor), torch.tensor(
                    query_labelmap).type(torch.LongTensor)

                query_labelmap = query_labelmap == query_label

                volume_prediction = []
                for i in range(0, len(query_volume), batch_size):
                    query_batch_x = query_volume[i: i + batch_size]
                    support_batch_x = support_volume[i: i + batch_size]

                    if cuda_available:
                        query_batch_x = query_batch_x.cuda(device)
                        support_batch_x = support_batch_x.cuda(device)

                    weights = model.conditioner(support_batch_x)
                    out = model.segmentor(query_batch_x, weights)

                    _, batch_output = torch.max(out, dim=1)
                    volume_prediction.append(batch_output)

                volume_prediction = torch.cat(volume_prediction)
                volume_dice_score = dice_score_binary(volume_prediction, query_labelmap.cuda(device), phase=mode)

                volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
                nifti_img = nib.MGHImage(np.squeeze(volume_prediction), np.eye(4))
                nib.save(nifti_img, os.path.join(prediction_path, volumes_query[vol_idx] + '_' + fold + str('.mgz')))

                nifti_img = nib.MGHImage(np.squeeze(query_volume.cpu().numpy()), np.eye(4))
                nib.save(nifti_img, os.path.join(prediction_path, volumes_query[vol_idx] + '_Input_' + str('.mgz')))

                # if logWriter:
                #     logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx],
                #                               vol_idx)
                volume_dice_score = volume_dice_score.cpu().numpy()
                volume_dice_score_list.append(volume_dice_score)

                print(volume_dice_score)

            dice_score_arr = np.asarray(volume_dice_score_list)
            avg_dice_score = np.mean(dice_score_arr)
            print('Query Label -> ' + str(query_label) + ' ' + str(avg_dice_score))
            all_query_dice_score_list.append(avg_dice_score)
        # class_dist = [dice_score_arr[:, c] for c in range(num_classes)]

        # if logWriter:
        #     logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
    print("DONE")

    return np.mean(all_query_dice_score_list)
