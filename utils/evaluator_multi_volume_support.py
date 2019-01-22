import os

import nibabel as nib
import numpy as np
import torch

import utils.common_utils as common_utils
import utils.data_utils as du
import torch.nn.functional as F
import matplotlib.pyplot as plt
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


def get_range(volume):
    batch, _, _ = volume.size()
    slice_with_class = torch.sum(volume.view(batch, -1), dim=1) > 10
    index = slice_with_class[:-1] - slice_with_class[1:] > 0
    seq = torch.Tensor(range(batch - 1))
    range_index = seq[index].type(torch.LongTensor)
    return range_index


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
    batch, _, _ = groud_truth.size()
    slice_with_class = torch.sum(groud_truth.view(batch, -1), dim=1) > 10
    index = slice_with_class[:-1] - slice_with_class[1:] > 0
    seq = torch.Tensor(range(batch - 1))
    range_index = seq[index].type(torch.LongTensor)
    groud_truth = groud_truth[slice_with_class]
    volume = volume[slice_with_class]
    condition_input = torch.cat((volume, groud_truth.unsqueeze(1)), dim=1)
    return condition_input, range_index.cpu().numpy()


def CV(samples_arr):
    eps = 0.0001
    threshold = 0.0001
    sample_size = samples_arr[0].size()
    total_pixels = sample_size[-1] * sample_size[-2]

    samples_arr = [sample.squeeze() for sample in samples_arr if
                   (sample.sum().item() / total_pixels) > threshold]

    if len(samples_arr) > 0:
        samples_arr = torch.cat(samples_arr).float().squeeze()
        std = torch.std(samples_arr).item()
        mean = torch.mean(samples_arr).item() + eps
        return std / mean
    else:
        return 1000000


def IOU_Single(sample1, sample2):
    eps = 0.0001
    sample1, sample2 = sample1.squeeze().byte(), sample2.squeeze().byte()
    intersection = sample1 & sample2
    union = sample1 | sample2

    numerator = intersection.sum().item()
    denominator = union.sum().item() + eps
    return numerator / denominator


def IoU(samples_arr, support, query, vol):
    # Hardcoded for 3 elements as of now
    eps = 0.0001
    threshold = 0.0001
    sample_size = samples_arr[0].size()
    total_pixels = sample_size[-1] * sample_size[-2]

    samples_arr = [sample.squeeze().byte() for sample in samples_arr if
                   (sample.sum().item() / total_pixels) > threshold]

    # plt.imsave(str(vol) + '_' + str(query) + '_' + str(support) + '_' + 'fig1', sample1.cpu().numpy())
    # plt.imsave(str(vol) + '_' + str(query) + '_' + str(support) + '_' + 'fig2', sample2.cpu().numpy())
    # plt.imsave(str(vol) + '_' + str(query) + '_' + str(support) + '_' + 'fig3', sample3.cpu().numpy())
    if len(samples_arr) > 0:
        intersection = torch.ones(samples_arr[0].size()).byte().cuda()
        union = torch.zeros(samples_arr[0].size()).byte().cuda()

        for sample in samples_arr:
            intersection = intersection & sample
            union = union | sample

        numerator = intersection.sum().item()
        denominator = union.sum().item() + eps

        return numerator / denominator
    else:
        return 0

    # inter = torch.ones(samples_arr[0].size()).cuda()
    # union = torch.zeros(samples_arr[0].size()).cuda()
    #
    #
    # for sample in samples_arr:
    #     inter = torch.sum(torch.mul(inter, sample.type(torch.cuda.FloatTensor)))
    #     union = torch.sum(union) + torch.sum(sample.type(torch.cuda.FloatTensor)) - inter
    # return -torch.div(inter, union + 0.0001).item()


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
    batch_size = 20
    Num_support = 15
    MC_samples = 10
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

            support_slices = []

            for i, file_path in enumerate(support_file_paths):
                # Loading support
                support_volume, support_labelmap, _, _ = du.load_and_preprocess(file_path,
                                                                                orientation=orientation,
                                                                                remap_config=remap_config)

                support_volume = support_volume if len(support_volume.shape) == 4 else support_volume[:, np.newaxis, :,
                                                                                       :]
                support_volume, support_labelmap = torch.tensor(support_volume).type(torch.FloatTensor), \
                                                   torch.tensor(support_labelmap).type(torch.LongTensor)

                support_volume, range_index = binarize_label(support_volume, support_labelmap, query_label)

                slice_gap_support = int(np.ceil(len(support_volume) / Num_support))

                support_slice_indexes = [i for i in range(0, len(support_volume), slice_gap_support)]

                if len(support_slice_indexes) < Num_support:
                    support_slice_indexes.append(len(support_volume) - 1)

                support_slices.extend([support_volume[idx] for idx in support_slice_indexes])

            for vol_idx, file_path in enumerate(query_file_paths):

                query_volume, query_labelmap, _, _ = du.load_and_preprocess(file_path,
                                                                            orientation=orientation,
                                                                            remap_config=remap_config)

                query_volume = query_volume if len(query_volume.shape) == 4 else query_volume[:, np.newaxis, :, :]
                query_volume, query_labelmap = torch.tensor(query_volume).type(torch.FloatTensor), \
                                               torch.tensor(query_labelmap).type(torch.LongTensor)

                query_labelmap = query_labelmap == query_label
                range_query = get_range(query_labelmap)
                query_volume = query_volume[range_query[0]: range_query[1] + 1]
                query_labelmap = query_labelmap[range_query[0]: range_query[1] + 1]

                slice_gap_query = int(np.ceil(len(query_volume) / Num_support))
                dice_per_batch = []
                batch_output_arr = []
                for support_slice_idx, i in enumerate(range(0, len(query_volume), slice_gap_query)):
                    query_batch_x = query_volume[i:i+slice_gap_query]
                    support_batch_x = support_volume[support_slice_idx].repeat(query_batch_x.size()[0], 1, 1, 1)

                    if cuda_available:
                        query_batch_x = query_batch_x.cuda(device)
                        support_batch_x = support_batch_x.cuda(device)

                        weights = model.conditioner(support_batch_x)
                        out = model.segmentor(query_batch_x, weights)

                        _, batch_output = torch.max(F.softmax(out, dim=1), dim=1)
                        batch_output_arr.append(batch_output)

                volume_output = torch.cat(batch_output_arr)
                volume_dice_score = dice_score_binary(volume_output, query_labelmap.cuda(device), phase=mode)
                volume_dice_score_list.append(volume_dice_score.item())
                print(str(file_path), volume_dice_score)

            dice_score_arr = np.asarray(volume_dice_score_list)
            avg_dice_score = np.median(dice_score_arr)
            print(volume_dice_score_list)
            print('Query Label -> ' + str(query_label) + ' ' + str(avg_dice_score))
            all_query_dice_score_list.append(avg_dice_score)

    print("DONE")

    return np.mean(all_query_dice_score_list)


def evaluate_dice_score_2view(model1_path,
                              model2_path,
                              num_classes,
                              query_labels,
                              data_dir,
                              query_txt_file,
                              support_txt_file,
                              remap_config,
                              orientation1,
                              prediction_path, device=0, logWriter=None, mode='eval', fold=None):
    print("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")
    print("Loading model => " + model1_path + " and " + model2_path)
    batch_size = 10

    with open(query_txt_file) as file_handle:
        volumes_query = file_handle.read().splitlines()

    # with open(support_txt_file) as file_handle:
    #     volumes_support = file_handle.read().splitlines()

    model1 = torch.load(model1_path)
    model2 = torch.load(model2_path)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        model1.cuda(device)
        model2.cuda(device)

    model1.eval()
    model2.eval()

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
                support_volume1, support_labelmap1, _, _ = du.load_and_preprocess(file_path,
                                                                                  orientation=orientation1,
                                                                                  remap_config=remap_config)
                support_volume2, support_labelmap2 = support_volume1.transpose((1, 2, 0)), support_labelmap1.transpose(
                    (1, 2, 0))

                support_volume1 = support_volume1 if len(support_volume1.shape) == 4 else support_volume1[:, np.newaxis,
                                                                                          :, :]
                support_volume2 = support_volume2 if len(support_volume2.shape) == 4 else support_volume2[:, np.newaxis,
                                                                                          :, :]

                support_volume1, support_labelmap1 = torch.tensor(support_volume1).type(
                    torch.FloatTensor), torch.tensor(
                    support_labelmap1).type(torch.LongTensor)
                support_volume2, support_labelmap2 = torch.tensor(support_volume2).type(
                    torch.FloatTensor), torch.tensor(
                    support_labelmap2).type(torch.LongTensor)
                support_volume1 = binarize_label(support_volume1, support_labelmap1, query_label)
                support_volume2 = binarize_label(support_volume2, support_labelmap2, query_label)

            for vol_idx, file_path in enumerate(query_file_paths):
                query_volume1, query_labelmap1, _, _ = du.load_and_preprocess(file_path,
                                                                              orientation=orientation1,
                                                                              remap_config=remap_config)
                query_volume2, query_labelmap2 = query_volume1.transpose((1, 2, 0)), query_labelmap1.transpose(
                    (1, 2, 0))

                query_volume1 = query_volume1 if len(query_volume1.shape) == 4 else query_volume1[:, np.newaxis, :, :]
                query_volume2 = query_volume2 if len(query_volume2.shape) == 4 else query_volume2[:, np.newaxis, :, :]

                query_volume1, query_labelmap1 = torch.tensor(query_volume1).type(torch.FloatTensor), torch.tensor(
                    query_labelmap1).type(torch.LongTensor)
                query_volume2, query_labelmap2 = torch.tensor(query_volume2).type(torch.FloatTensor), torch.tensor(
                    query_labelmap2).type(torch.LongTensor)

                query_labelmap1 = query_labelmap1 == query_label
                query_labelmap2 = query_labelmap2 == query_label

                # Evaluate for orientation 1
                support_batch_x = []
                k = 2
                volume_prediction1 = []
                for i in range(0, len(query_volume1), batch_size):
                    query_batch_x = query_volume1[i: i + batch_size]
                    if k % 2 == 0:
                        support_batch_x = support_volume1[i: i + batch_size]
                    sz = query_batch_x.size()
                    support_batch_x = support_batch_x[batch_size - 1].repeat(sz[0], 1, 1, 1)
                    k += 1
                    if cuda_available:
                        query_batch_x = query_batch_x.cuda(device)
                        support_batch_x = support_batch_x.cuda(device)

                    weights = model1.conditioner(support_batch_x)
                    out = model1.segmentor(query_batch_x, weights)

                    # _, batch_output = torch.max(F.softmax(out, dim=1), dim=1)
                    volume_prediction1.append(out)

                # Evaluate for orientation 2
                support_batch_x = []
                k = 2
                volume_prediction2 = []
                for i in range(0, len(query_volume2), batch_size):
                    query_batch_x = query_volume2[i: i + batch_size]
                    if k % 2 == 0:
                        support_batch_x = support_volume2[i: i + batch_size]
                    sz = query_batch_x.size()
                    support_batch_x = support_batch_x[batch_size - 1].repeat(sz[0], 1, 1, 1)
                    k += 1
                    if cuda_available:
                        query_batch_x = query_batch_x.cuda(device)
                        support_batch_x = support_batch_x.cuda(device)

                    weights = model2.conditioner(support_batch_x)
                    out = model2.segmentor(query_batch_x, weights)
                    volume_prediction2.append(out)

                volume_prediction1 = torch.cat(volume_prediction1)
                volume_prediction2 = torch.cat(volume_prediction2)
                volume_prediction = 0.5 * volume_prediction1 + 0.5 * volume_prediction2.permute(3, 1, 0, 2)
                _, batch_output = torch.max(F.softmax(volume_prediction, dim=1), dim=1)
                volume_dice_score = dice_score_binary(batch_output, query_labelmap1.cuda(device), phase=mode)

                batch_output = (batch_output.cpu().numpy()).astype('float32')
                nifti_img = nib.MGHImage(np.squeeze(batch_output), np.eye(4))
                nib.save(nifti_img, os.path.join(prediction_path, volumes_query[vol_idx] + '_' + fold + str('.mgz')))

                # # Save Input
                # nifti_img = nib.MGHImage(np.squeeze(query_volume1.cpu().numpy()), np.eye(4))
                # nib.save(nifti_img, os.path.join(prediction_path, volumes_query[vol_idx] + '_Input_' + str('.mgz')))
                # # # Condition Input
                # nifti_img = nib.MGHImage(np.squeeze(support_volume1.cpu().numpy()), np.eye(4))
                # nib.save(nifti_img, os.path.join(prediction_path, volumes_query[vol_idx] + '_CondInput_' + str('.mgz')))
                # # # Cond GT
                # nifti_img = nib.MGHImage(np.squeeze(support_labelmap1.cpu().numpy()).astype('float32'), np.eye(4))
                # nib.save(nifti_img,
                #          os.path.join(prediction_path, volumes_query[vol_idx] + '_CondInputGT_' + str('.mgz')))
                # # # # Save Ground Truth
                # nifti_img = nib.MGHImage(np.squeeze(query_labelmap1.cpu().numpy()), np.eye(4))
                # nib.save(nifti_img, os.path.join(prediction_path, volumes_query[vol_idx] + '_GT_' + str('.mgz')))

                # if logWriter:
                #     logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx],
                #                               vol_idx)
                volume_dice_score = volume_dice_score.cpu().numpy()
                volume_dice_score_list.append(volume_dice_score)

                print(volume_dice_score)

            dice_score_arr = np.asarray(volume_dice_score_list)
            avg_dice_score = np.median(dice_score_arr)
            print('Query Label -> ' + str(query_label) + ' ' + str(avg_dice_score))
            all_query_dice_score_list.append(avg_dice_score)
        # class_dist = [dice_score_arr[:, c] for c in range(num_classes)]

        # if logWriter:
        #     logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
    print("DONE")

    return np.mean(all_query_dice_score_list)


def evaluate_dice_score_3view(model1_path,
                              model2_path,
                              model3_path,
                              num_classes,
                              query_labels,
                              data_dir,
                              query_txt_file,
                              support_txt_file,
                              remap_config,
                              orientation1,
                              prediction_path, device=0, logWriter=None, mode='eval', fold=None):
    print("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")
    print("Loading model => " + model1_path + " and " + model2_path)
    batch_size = 10

    with open(query_txt_file) as file_handle:
        volumes_query = file_handle.read().splitlines()

    # with open(support_txt_file) as file_handle:
    #     volumes_support = file_handle.read().splitlines()

    model1 = torch.load(model1_path)
    model2 = torch.load(model2_path)
    model3 = torch.load(model3_path)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        model1.cuda(device)
        model2.cuda(device)
        model3.cuda(device)

    model1.eval()
    model2.eval()
    model3.eval()

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
                support_volume1, support_labelmap1, _, _ = du.load_and_preprocess(file_path,
                                                                                  orientation=orientation1,
                                                                                  remap_config=remap_config)
                support_volume2, support_labelmap2 = support_volume1.transpose((1, 2, 0)), support_labelmap1.transpose(
                    (1, 2, 0))

                support_volume3, support_labelmap3 = support_volume1.transpose((2, 0, 1)), support_labelmap1.transpose(
                    (2, 0, 1))

                support_volume1 = support_volume1 if len(support_volume1.shape) == 4 else support_volume1[:, np.newaxis,
                                                                                          :, :]
                support_volume2 = support_volume2 if len(support_volume2.shape) == 4 else support_volume2[:, np.newaxis,
                                                                                          :, :]

                support_volume3 = support_volume3 if len(support_volume3.shape) == 4 else support_volume3[:, np.newaxis,
                                                                                          :, :]

                support_volume1, support_labelmap1 = torch.tensor(support_volume1).type(
                    torch.FloatTensor), torch.tensor(
                    support_labelmap1).type(torch.LongTensor)
                support_volume2, support_labelmap2 = torch.tensor(support_volume2).type(
                    torch.FloatTensor), torch.tensor(
                    support_labelmap2).type(torch.LongTensor)
                support_volume3, support_labelmap3 = torch.tensor(support_volume3).type(
                    torch.FloatTensor), torch.tensor(
                    support_labelmap3).type(torch.LongTensor)

                support_volume1 = binarize_label(support_volume1, support_labelmap1, query_label)
                support_volume2 = binarize_label(support_volume2, support_labelmap2, query_label)
                support_volume3 = binarize_label(support_volume3, support_labelmap3, query_label)

            for vol_idx, file_path in enumerate(query_file_paths):
                query_volume1, query_labelmap1, _, _ = du.load_and_preprocess(file_path,
                                                                              orientation=orientation1,
                                                                              remap_config=remap_config)
                query_volume2, query_labelmap2 = query_volume1.transpose((1, 2, 0)), query_labelmap1.transpose(
                    (1, 2, 0))
                query_volume3, query_labelmap3 = query_volume1.transpose((2, 0, 1)), query_labelmap1.transpose(
                    (2, 0, 1))

                query_volume1 = query_volume1 if len(query_volume1.shape) == 4 else query_volume1[:, np.newaxis, :, :]
                query_volume2 = query_volume2 if len(query_volume2.shape) == 4 else query_volume2[:, np.newaxis, :, :]
                query_volume3 = query_volume3 if len(query_volume3.shape) == 4 else query_volume3[:, np.newaxis, :, :]

                query_volume1, query_labelmap1 = torch.tensor(query_volume1).type(torch.FloatTensor), torch.tensor(
                    query_labelmap1).type(torch.LongTensor)
                query_volume2, query_labelmap2 = torch.tensor(query_volume2).type(torch.FloatTensor), torch.tensor(
                    query_labelmap2).type(torch.LongTensor)
                query_volume3, query_labelmap3 = torch.tensor(query_volume3).type(torch.FloatTensor), torch.tensor(
                    query_labelmap3).type(torch.LongTensor)

                query_labelmap1 = query_labelmap1 == query_label
                # query_labelmap2 = query_labelmap2 == query_label
                # query_labelmap3 = query_labelmap3 == query_label

                # Evaluate for orientation 1
                support_batch_x = []
                k = 2
                volume_prediction1 = []
                for i in range(0, len(query_volume1), batch_size):
                    query_batch_x = query_volume1[i: i + batch_size]
                    if k % 2 == 0:
                        support_batch_x = support_volume1[i: i + batch_size]
                    sz = query_batch_x.size()
                    support_batch_x = support_batch_x[batch_size - 1].repeat(sz[0], 1, 1, 1)
                    k += 1
                    if cuda_available:
                        query_batch_x = query_batch_x.cuda(device)
                        support_batch_x = support_batch_x.cuda(device)

                    weights = model1.conditioner(support_batch_x)
                    out = model1.segmentor(query_batch_x, weights)

                    # _, batch_output = torch.max(F.softmax(out, dim=1), dim=1)
                    volume_prediction1.append(out)

                # Evaluate for orientation 2
                support_batch_x = []
                k = 2
                volume_prediction2 = []
                for i in range(0, len(query_volume2), batch_size):
                    query_batch_x = query_volume2[i: i + batch_size]
                    if k % 2 == 0:
                        support_batch_x = support_volume2[i: i + batch_size]
                    sz = query_batch_x.size()
                    support_batch_x = support_batch_x[batch_size - 1].repeat(sz[0], 1, 1, 1)
                    k += 1
                    if cuda_available:
                        query_batch_x = query_batch_x.cuda(device)
                        support_batch_x = support_batch_x.cuda(device)

                    weights = model2.conditioner(support_batch_x)
                    out = model2.segmentor(query_batch_x, weights)
                    volume_prediction2.append(out)

                # Evaluate for orientation 3
                support_batch_x = []
                k = 2
                volume_prediction3 = []
                for i in range(0, len(query_volume3), batch_size):
                    query_batch_x = query_volume3[i: i + batch_size]
                    if k % 2 == 0:
                        support_batch_x = support_volume3[i: i + batch_size]
                    sz = query_batch_x.size()
                    support_batch_x = support_batch_x[batch_size - 1].repeat(sz[0], 1, 1, 1)
                    k += 1
                    if cuda_available:
                        query_batch_x = query_batch_x.cuda(device)
                        support_batch_x = support_batch_x.cuda(device)

                    weights = model3.conditioner(support_batch_x)
                    out = model3.segmentor(query_batch_x, weights)
                    volume_prediction3.append(out)

                volume_prediction1 = torch.cat(volume_prediction1)
                volume_prediction2 = torch.cat(volume_prediction2)
                volume_prediction3 = torch.cat(volume_prediction3)
                volume_prediction = 0.33 * F.softmax(volume_prediction1, dim=1) + 0.33 * F.softmax(
                    volume_prediction2.permute(3, 1, 0, 2), dim=1) + 0.33 * F.softmax(
                    volume_prediction3.permute(2, 1, 3, 0), dim=1)
                _, batch_output = torch.max(volume_prediction, dim=1)
                volume_dice_score = dice_score_binary(batch_output, query_labelmap1.cuda(device), phase=mode)

                batch_output = (batch_output.cpu().numpy()).astype('float32')
                nifti_img = nib.MGHImage(np.squeeze(batch_output), np.eye(4))
                nib.save(nifti_img, os.path.join(prediction_path, volumes_query[vol_idx] + '_' + fold + str('.mgz')))

                # # Save Input
                # nifti_img = nib.MGHImage(np.squeeze(query_volume1.cpu().numpy()), np.eye(4))
                # nib.save(nifti_img, os.path.join(prediction_path, volumes_query[vol_idx] + '_Input_' + str('.mgz')))
                # # # Condition Input
                # nifti_img = nib.MGHImage(np.squeeze(support_volume1.cpu().numpy()), np.eye(4))
                # nib.save(nifti_img, os.path.join(prediction_path, volumes_query[vol_idx] + '_CondInput_' + str('.mgz')))
                # # # Cond GT
                # nifti_img = nib.MGHImage(np.squeeze(support_labelmap1.cpu().numpy()).astype('float32'), np.eye(4))
                # nib.save(nifti_img,
                #          os.path.join(prediction_path, volumes_query[vol_idx] + '_CondInputGT_' + str('.mgz')))
                # # # # Save Ground Truth
                # nifti_img = nib.MGHImage(np.squeeze(query_labelmap1.cpu().numpy()), np.eye(4))
                # nib.save(nifti_img, os.path.join(prediction_path, volumes_query[vol_idx] + '_GT_' + str('.mgz')))

                # if logWriter:
                #     logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx],
                #                               vol_idx)
                volume_dice_score = volume_dice_score.cpu().numpy()
                volume_dice_score_list.append(volume_dice_score)

                print(volume_dice_score)

            dice_score_arr = np.asarray(volume_dice_score_list)
            avg_dice_score = np.median(dice_score_arr)
            print('Query Label -> ' + str(query_label) + ' ' + str(avg_dice_score))
            all_query_dice_score_list.append(avg_dice_score)
        # class_dist = [dice_score_arr[:, c] for c in range(num_classes)]

        # if logWriter:
        #     logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
    print("DONE")

    return np.mean(all_query_dice_score_list)
