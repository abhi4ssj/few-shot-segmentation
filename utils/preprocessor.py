import numpy as np

ORIENTATION = {
    'coronal': "COR",
    'axial': "AXI",
    'sagital': "SAG"
}


def rotate_orientation(volume_data, volume_label, orientation=ORIENTATION['coronal']):
    if orientation == ORIENTATION['coronal']:
        return volume_data.transpose((2, 0, 1)), volume_label.transpose((2, 0, 1))
    elif orientation == ORIENTATION['axial']:
        return volume_data.transpose((1, 2, 0)), volume_label.transpose((1, 2, 0))
    elif orientation == ORIENTATION['sagital']:
        return volume_data, volume_label
    else:
        raise ValueError("Invalid value for orientation. Pleas see help")


def estimate_weights_mfb(labels):
    class_weights = np.zeros_like(labels)
    unique, counts = np.unique(labels, return_counts=True)
    median_freq = np.median(counts)
    weights = np.zeros(len(unique))
    for i, label in enumerate(unique):
        class_weights += (median_freq // counts[i]) * np.array(labels == label)
        try:
            weights[int(label)] = median_freq // counts[i]
        except IndexError as e:
            print("Exception in processing")
            continue

    grads = np.gradient(labels)
    edge_weights = (grads[0] ** 2 + grads[1] ** 2) > 0
    class_weights += 2 * edge_weights
    return class_weights, weights


def remap_labels(labels, remap_config):
    """
    Function to remap the label values into the desired range of algorithm
    """
    if remap_config == 'FS':
        label_list = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50,
                      51, 52, 53, 54, 58, 60]
    elif remap_config == 'Neo':
        labels[(labels >= 100) & (labels % 2 == 0)] = 210
        labels[(labels >= 100) & (labels % 2 == 1)] = 211
        label_list = [45, 211, 52, 50, 41, 39, 60, 37, 58, 56, 4, 11, 35, 48, 32, 46, 30, 62, 44, 210, 51, 49, 40, 38,
                      59, 36, 57, 55, 47, 31, 23, 61]

    elif remap_config == 'WholeBody':
        label_list = [1, 2, 7, 8, 9, 13, 14, 17, 18]

    elif remap_config == 'brain_fewshot':
        labels[(labels >= 100) & (labels % 2 == 0)] = 210
        labels[(labels >= 100) & (labels % 2 == 1)] = 211
        label_list = [[210, 211], [45, 44], [52, 51], [35], [39, 41, 40, 38], [36, 37, 57, 58, 60, 59, 56, 55]]
    else:
        raise ValueError("Invalid argument value for remap config, only valid options are FS and Neo")

    new_labels = np.zeros_like(labels)

    k = isinstance(label_list[0], list)

    if not k:
        for i, label in enumerate(label_list):
            label_present = np.zeros_like(labels)
            label_present[labels == label] = 1
            new_labels = new_labels + (i + 1) * label_present
    else:
        for i, label in enumerate(label_list):
            label_present = np.zeros_like(labels)
            for j in label:
                label_present[labels == j] = 1
            new_labels = new_labels + (i + 1) * label_present
    return new_labels


def reduce_slices(data, labels, skip_Frame=40):
    """
    This function removes the useless black slices from the start and end. And then selects every even numbered frame.
    """
    no_slices, H, W = data.shape
    mask_vector = np.zeros(no_slices, dtype=int)
    mask_vector[::2], mask_vector[1::2] = 1, 0
    mask_vector[:skip_Frame], mask_vector[-skip_Frame:-1] = 0, 0

    data_reduced = np.compress(mask_vector, data, axis=0).reshape(-1, H, W)
    labels_reduced = np.compress(mask_vector, labels, axis=0).reshape(-1, H, W)

    return data_reduced, labels_reduced


def remove_black(data, labels):
    clean_data, clean_labels = [], []
    for i, frame in enumerate(labels):
        unique, counts = np.unique(frame, return_counts=True)
        if counts[0] / sum(counts) < .99:
            clean_labels.append(frame)
            clean_data.append(data[i])
    return np.array(clean_data), np.array(clean_labels)
