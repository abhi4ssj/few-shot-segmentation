import os

import nibabel as nib
import numpy as np
import torch

import utils.common_utils as common_utils
import utils.data_utils as du


def dice_confusion_matrix(vol_output, ground_truth, num_classes):
    dice_cm = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        for j in range(num_classes):
            Pred = (vol_output == j).float()
            inter = torch.sum(torch.mul(GT, Pred))
            union = torch.sum(GT) + torch.sum(Pred) + 0.0001
            dice_cm[i, j] = 2 * torch.div(inter, union)
    avg_dice = torch.mean(torch.diagflat(dice_cm))
    return avg_dice, dice_cm

def dice_score_perclass(vol_output, ground_truth, num_classes):
    dice_perclass = torch.zeros(num_classes)
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred))
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
    return dice_perclass


def evaluate_dice_score(model_path, num_classes, data_dir, label_dir, volumes_txt_file, remap_config, orientation,
                        prediction_path, device=0, logWriter=None):
    print("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")

    batch_size = 20

    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()

    model = torch.load(model_path)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        model.cuda(device)

    model.eval()

    common_utils.create_if_not(prediction_path)
    volume_dice_score_list = []
    print("Evaluating now...")
    file_paths = du.load_file_paths(data_dir, label_dir, volumes_txt_file)
    with torch.no_grad():
        for vol_idx, file_path in enumerate(file_paths):
            volume, labelmap = du.load_and_preprocess(file_path, orientation=orientation, remap_config=remap_config)

            volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
            volume, labelmap = torch.tensor(volume).type(torch.FloatTensor), torch.tensor(labelmap).type(
                torch.LongTensor)

            volume_prediction = []
            for i in range(0, len(volume), batch_size):
                batch_x, batch_y = volume[i: i + batch_size], labelmap[i:i + batch_size]
                if cuda_available:
                    batch_x = batch_x.cuda(device)
                out = model(batch_x)
                _, batch_output = torch.max(out, dim=1)
                volume_prediction.append(batch_output)

            volume_prediction = torch.cat(volume_prediction)
            volume_dice_score = dice_score_perclass(volume_prediction, labelmap.cuda(device), num_classes)

            volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
            nifti_img = nib.MGHImage(np.squeeze(volume_prediction), np.eye(4))
            nib.save(nifti_img, os.path.join(prediction_path, volumes_to_use[vol_idx] + str('.mgz')))
            if logWriter:
                logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx], vol_idx)

            volume_dice_score = volume_dice_score.cpu().numpy()
            volume_dice_score_list.append(volume_dice_score)
            print(volume_dice_score, np.mean(volume_dice_score))
        dice_score_arr = np.asarray(volume_dice_score_list)
        avg_dice_score = np.mean(dice_score_arr)
        print("Mean of dice score : " + str(avg_dice_score))
        class_dist = [dice_score_arr[:, c] for c in range(num_classes)]

        if logWriter:
            logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
    print("DONE")

    return avg_dice_score, class_dist
