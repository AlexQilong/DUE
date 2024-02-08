import glob
import os
import re
import csv
import cv2
import numpy as np
import pandas as pd
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


def get_all_npy(f_path):
    """Get the .npy files in a folder"""
    subdirs = [x[0] for x in os.walk(f_path)]
    maxlen = len(max(subdirs, key=len))
    maxsubdirs = [x for x in subdirs if len(x) == maxlen]
    for p in maxsubdirs:
        fs = glob.glob(os.path.join(p + "/", "*.npy"))
    
    return sorted(fs)


def get_patient_id(mode, num_samples=None, random_seed=None):
    """Get patient ID"""
    assert mode in ['train', 'val', 'test']
    # Read CSV file into a DataFrame
    csv_file_path = f'/your/data/path/{mode}.csv'
    df = pd.read_csv(csv_file_path)
    patient_ids = df.values.tolist()
    # Sample k IDs
    patient_ids = pd.Series(patient_ids).sample(n=num_samples, random_state=random_seed).tolist()

    return patient_ids


def balance_lidc_data(mode, folder, num_samples=None, random_seed=None):
    """Return .npy files"""
        
    patient_ids_of_interest = get_patient_id(mode)

    data_path = f'/your/data/path/{folder}'
    data_all = get_all_npy(data_path)
        
    data_return = []
    for data in data_all:
        current_id = [data.split('/')[-1][:14]]

        if current_id in patient_ids_of_interest:
            data_return.append(np.load(data))
            # 1 sample per patient
            patient_ids_of_interest.remove(current_id)
    
    return data_return


def balance_pancreas_data(mode, folder):

    assert mode == 'train'

    data_path = f'/your/data/path/{folder}'
    data_all = get_all_npy(data_path)
    
    data_return = []
    for data in data_all:
        data_return.append(np.load(data))

    return data_return


def imbalance_lidc_data(mode, folder):
    patient_ids_of_interest = get_patient_id(mode)

    data_path = f'/your/data/path/{folder}'
    data_all = get_all_npy(data_path)
        
    data_return = []
    for data in data_all:
        current_id = [data.split('/')[-1][:14]]

        if current_id in patient_ids_of_interest:
            data_return.append(np.load(data))
    
    return data_return


def imbalance_pancreas_data(mode, folder):
    
    assert mode == 'val' or mode == 'test'

    data_path = f'/your/data/path/{folder}'
    data_all = get_all_npy(data_path)
    
    train_ratio = 0.125
    val_ratio = 0.175
    test_ratio = 0.7

    if mode == 'val':
        start_id = int(train_ratio * len(data_all))
        end_id = int(start_id + val_ratio * len(data_all))
        data_split = data_all[start_id:end_id]

    elif mode == 'test':
        start_id = int((1 - test_ratio) * len(data_all))
        data_split = data_all[start_id:]

    data_return = []
    for data in data_split:
        data_return.append(np.load(data))

    return data_return


def get_X_y(mode, C, H, W, D):

    pos_samples = balance_lidc_data(mode, 'pos')
    pos_labels = np.array([1 for _ in range(len(pos_samples))])
    pos_samples = np.array([preprocess_volume(volume, (H, W, D)) for volume in pos_samples])
    pos_samples = pos_samples.reshape(len(pos_samples), C, H, W, D)
    pos_samples = np.transpose(pos_samples, (0, 1, 4, 2, 3))
    
    neg_samples = balance_lidc_data(mode, 'neg')
    neg_labels = np.array([0 for _ in range(len(neg_samples))])
    neg_samples = np.array([preprocess_volume(volume, (H, W, D)) for volume in neg_samples])
    neg_samples = neg_samples.reshape(len(neg_samples), C, H, W, D)
    neg_samples = np.transpose(neg_samples, (0, 1, 4, 2, 3))

    X = np.concatenate((pos_samples, neg_samples), axis=0)
    y = np.concatenate((pos_labels, neg_labels), axis=0)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    return X, y


def get_X_y_pan(mode, C, H, W, D):

    pos_samples = balance_pancreas_data(mode, 'pos')
    pos_labels = np.array([1 for _ in range(len(pos_samples))])
    pos_samples = np.array([preprocess_volume(volume, (H, W, D)) for volume in pos_samples])
    pos_samples = pos_samples.reshape(len(pos_samples), C, H, W, D)
    pos_samples = np.transpose(pos_samples, (0, 1, 4, 2, 3))
    
    neg_samples = balance_pancreas_data(mode, 'neg')
    neg_labels = np.array([0 for _ in range(len(neg_samples))])
    neg_samples = np.array([preprocess_volume(volume, (H, W, D)) for volume in neg_samples])
    neg_samples = neg_samples.reshape(len(neg_samples), C, H, W, D)
    neg_samples = np.transpose(neg_samples, (0, 1, 4, 2, 3))

    X = np.concatenate((pos_samples, neg_samples), axis=0)
    y = np.concatenate((pos_labels, neg_labels), axis=0)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    return X, y


def get_X_y_imbal(mode, C, H, W, D):

    pos_samples = imbalance_lidc_data(mode, 'pos')
    neg_samples = imbalance_lidc_data(mode, 'neg')
    
    # generate labels
    pos_labels = np.array([1 for _ in range(len(pos_samples))])
    neg_labels = np.array([0 for _ in range(len(neg_samples))])

    pos_samples = np.array([preprocess_volume(volume, (H, W, D)) for volume in pos_samples])
    pos_samples = pos_samples.reshape(len(pos_samples), C, H, W, D)
    pos_samples = np.transpose(pos_samples, (0, 1, 4, 2, 3))
    
    neg_samples = np.array([preprocess_volume(volume, (H, W, D)) for volume in neg_samples])
    neg_samples = neg_samples.reshape(len(neg_samples), C, H, W, D)
    neg_samples = np.transpose(neg_samples, (0, 1, 4, 2, 3))

    X = np.concatenate((pos_samples, neg_samples), axis=0)
    y = np.concatenate((pos_labels, neg_labels), axis=0)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    return X, y, len(neg_samples)


def get_X_y_pan_imbal(mode, C, H, W, D):

    pos_samples = imbalance_pancreas_data(mode, 'pos')
    neg_samples = imbalance_pancreas_data(mode, 'neg')
    
    # generate labels
    pos_labels = np.array([1 for _ in range(len(pos_samples))])
    neg_labels = np.array([0 for _ in range(len(neg_samples))])

    pos_samples = np.array([preprocess_volume(volume, (H, W, D)) for volume in pos_samples])
    pos_samples = pos_samples.reshape(len(pos_samples), C, H, W, D)
    pos_samples = np.transpose(pos_samples, (0, 1, 4, 2, 3))
    
    neg_samples = np.array([preprocess_volume(volume, (H, W, D)) for volume in neg_samples])
    neg_samples = neg_samples.reshape(len(neg_samples), C, H, W, D)
    neg_samples = np.transpose(neg_samples, (0, 1, 4, 2, 3))

    X = np.concatenate((pos_samples, neg_samples), axis=0)
    y = np.concatenate((pos_labels, neg_labels), axis=0)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    return X, y, len(neg_samples)


def preprocess_mask(mask, target_shape):
    """Preprocess the binary mask"""
    mask = torch.from_numpy(mask)
    # resize
    mask = torch.unsqueeze(mask, 0)
    mask = torch.unsqueeze(mask, 0)
    mask_resized = resize_mask(mask, target_shape)

    return mask_resized.numpy()


def get_MASK_org_depth(mode, C, H, W, D):

    pos_masks = balance_lidc_data(mode, 'cmask')
    org_depth = np.array([get_vol_depth(mask) for mask in pos_masks])
    
    pos_masks = np.array([preprocess_mask(mask, (H, W, D)) for mask in pos_masks])
    pos_masks = pos_masks.reshape(len(pos_masks), C, H, W, D)
    pos_masks = np.transpose(pos_masks, (0, 1, 4, 2, 3))

    neg_labels = np.array([0 for _ in range(len(pos_masks))])
    # depth for neg samples are 0
    org_depth = np.concatenate((org_depth, neg_labels), axis=0)
    
    neg_masks = np.negative(np.ones(shape=(len(neg_labels), C, D, H, W), dtype=np.int8))
    MASK = np.concatenate((pos_masks, neg_masks), axis=0)

    MASK = torch.from_numpy(MASK).long()
    org_depth = torch.from_numpy(org_depth).long()
    
    return MASK, org_depth


def get_MASK_org_depth_pan(mode, C, H, W, D):

    pos_masks = balance_pancreas_data(mode, 'cmask')
    org_depth = np.array([get_vol_depth(mask) for mask in pos_masks])
    
    pos_masks = np.array([preprocess_mask(mask, (H, W, D)) for mask in pos_masks])
    pos_masks = pos_masks.reshape(len(pos_masks), C, H, W, D)
    pos_masks = np.transpose(pos_masks, (0, 1, 4, 2, 3))

    neg_labels = np.array([0 for _ in range(len(pos_masks))])
    # depth for neg samples are 0
    org_depth = np.concatenate((org_depth, neg_labels), axis=0)
    
    neg_masks = np.negative(np.ones(shape=(len(neg_labels), C, D, H, W), dtype=np.int8))
    MASK = np.concatenate((pos_masks, neg_masks), axis=0)

    MASK = torch.from_numpy(MASK).long()
    org_depth = torch.from_numpy(org_depth).long()
    
    return MASK, org_depth


def get_MASK_org_depth_imbal(mode, neg_num, C, H, W, D):
    
    pos_masks = imbalance_lidc_data(mode, 'cmask')
    org_depth = np.array([get_vol_depth(mask) for mask in pos_masks])
    
    pos_masks = np.array([preprocess_mask(mask, (H, W, D)) for mask in pos_masks])
    pos_masks = pos_masks.reshape(len(pos_masks), C, H, W, D)
    pos_masks = np.transpose(pos_masks, (0, 1, 4, 2, 3))

    neg_labels = np.array([0 for _ in range(neg_num)])
    # depth for neg samples are 0
    org_depth = np.concatenate((org_depth, neg_labels), axis=0)
    
    neg_masks = np.negative(np.ones(shape=(len(neg_labels), C, D, H, W), dtype=np.int8))
    MASK = np.concatenate((pos_masks, neg_masks), axis=0)

    MASK = torch.from_numpy(MASK).long()
    org_depth = torch.from_numpy(org_depth).long()
    
    return MASK, org_depth


def get_MASK_org_depth_pan_imbal(mode, neg_num, C, H, W, D):
    
    pos_masks = imbalance_pancreas_data(mode, 'cmask')
    org_depth = np.array([get_vol_depth(mask) for mask in pos_masks])
    
    pos_masks = np.array([preprocess_mask(mask, (H, W, D)) for mask in pos_masks])
    pos_masks = pos_masks.reshape(len(pos_masks), C, H, W, D)
    pos_masks = np.transpose(pos_masks, (0, 1, 4, 2, 3))

    neg_labels = np.array([0 for _ in range(neg_num)])
    # depth for neg samples are 0
    org_depth = np.concatenate((org_depth, neg_labels), axis=0)
    
    neg_masks = np.negative(np.ones(shape=(len(neg_labels), C, D, H, W), dtype=np.int8))
    MASK = np.concatenate((pos_masks, neg_masks), axis=0)

    MASK = torch.from_numpy(MASK).long()
    org_depth = torch.from_numpy(org_depth).long()
    
    return MASK, org_depth


def get_bal_lidc_dataset(mode, C, H, W, D):
    """Return TensorDataset"""

    X, y = get_X_y(mode, C, H, W, D)
    MASK, org_depth = get_MASK_org_depth(mode, C, H, W, D)

    if mode == 'train':
        return torch.utils.data.TensorDataset(X, y, MASK, org_depth)
    else:  # val/test  
        return torch.utils.data.TensorDataset(X, y, MASK)


def get_bal_pancreas_dataset(mode, C, H, W, D):
    """Return TensorDataset"""

    X, y = get_X_y_pan(mode, C, H, W, D)
    MASK, org_depth = get_MASK_org_depth_pan(mode, C, H, W, D)

    if mode == 'train':
        return torch.utils.data.TensorDataset(X, y, MASK, org_depth)
    else:  # val/test  
        return torch.utils.data.TensorDataset(X, y, MASK)


def get_lidc_dataset_imbal(mode, C, H, W, D):
    """Return TensorDataset"""

    X, y, neg_num = get_X_y_imbal(mode, C, H, W, D)
    MASK, org_depth = get_MASK_org_depth_imbal(mode, neg_num, C, H, W, D)

    if mode == 'train':
        return torch.utils.data.TensorDataset(X, y, MASK, org_depth)
    else:  # val/test  
        return torch.utils.data.TensorDataset(X, y, MASK)


def get_pancreas_dataset_imbal(mode, C, H, W, D):
    """Return TensorDataset"""

    X, y, neg_num = get_X_y_pan_imbal(mode, C, H, W, D)
    MASK, org_depth = get_MASK_org_depth_pan_imbal(mode, neg_num, C, H, W, D)

    if mode == 'train':
        return torch.utils.data.TensorDataset(X, y, MASK, org_depth)
    else:  # val/test  
        return torch.utils.data.TensorDataset(X, y, MASK)


def natural_sort_key(s):
    # Split the file name into parts using '_' and '.'
    parts = s.absolute().as_posix().split('_')
    if len(parts) >= 2:
        # Extract the characters between '_' and '.'
        new_name = parts[1].split('.')[0]

    return int(new_name)


def get_interp_annotation(C, H, W, D):
    from pathlib import Path
    import os

    interp_examples = []

    frames_dir = '/your/data/path/'
    frames_path = Path(frames_dir).absolute()

    frames_folders = os.listdir(frames_path)
    frames_folders = [frames_path.joinpath(s) for s in frames_folders]

    for folder in frames_folders:
        
        all_samples = sorted(list(folder.glob('*')), key=natural_sort_key)
        
        for sample in all_samples:

            sample_path = sample.absolute().as_posix()
            interp_examples.append(np.load(sample_path))
            
    interp_examples = np.array(interp_examples)
    interp_examples = interp_examples.reshape(len(interp_examples), C, D, H, W)

    neg_masks = np.negative(np.ones(shape=(len(interp_examples), C, D, H, W), dtype=np.float16))
    
    MASK = np.concatenate((interp_examples, neg_masks), axis=0)
    MASK = torch.from_numpy(MASK).float()

    return MASK


def get_interp_annotation_pan(C, H, W, D):
    from pathlib import Path
    import os

    interp_examples = []

    frames_dir = '/your/data/path/'
    frames_path = Path(frames_dir).absolute()

    frames_folders = os.listdir(frames_path)
    frames_folders = [frames_path.joinpath(s) for s in frames_folders]

    for folder in frames_folders:
        
        all_samples = sorted(list(folder.glob('*')), key=natural_sort_key)
        
        for sample in all_samples:

            sample_path = sample.absolute().as_posix()
            interp_examples.append(np.load(sample_path))
            
    interp_examples = np.array(interp_examples)
    interp_examples = interp_examples.reshape(len(interp_examples), C, D, H, W)

    neg_masks = np.negative(np.ones(shape=(len(interp_examples), C, D, H, W), dtype=np.float16))
    
    MASK = np.concatenate((interp_examples, neg_masks), axis=0)
    MASK = torch.from_numpy(MASK).float()

    return MASK


def get_lidc_train_due(mode, C, H, W, D):
    """Return TensorDataset"""
    assert mode == 'train'

    X, y = get_X_y(mode, C, H, W, D)
    MASK, org_depths = get_MASK_org_depth(mode, C, H, W, D)
    diffu_interp = get_interp_annotation(C, H, W, D)

    return torch.utils.data.TensorDataset(X, y, MASK, org_depths, diffu_interp)


def get_pancreas_train_due(mode, C, H, W, D):
    """Return TensorDataset"""
    assert mode == 'train'

    X, y = get_X_y_pan(mode, C, H, W, D)
    MASK, org_depths = get_MASK_org_depth_pan(mode, C, H, W, D)
    diffu_interp = get_interp_annotation_pan(C, H, W, D)

    return torch.utils.data.TensorDataset(X, y, MASK, org_depths, diffu_interp)


def resize_mask(mask, target_shape):
    """Resize the binary mask to target_shape"""
    mask_resized = F.interpolate(
        # convert bool to int
        (mask > 0).type(torch.uint8),
        size=target_shape,
        mode="nearest"
    )
    
    return mask_resized


def get_mask_blocks(gt_masks, target_shape, thickness=3):
    """Resize masks slice by slice"""
    gt_masks_resized = []
    for mask in gt_masks:  # gt_masks: shape = (num, H, W, D)
        depth = len(mask[0][0])
        # num_blocks = depth // thickness
        num_blocks = (depth - 2) // 1
        if num_blocks > 0:
            mask = torch.Tensor(mask)
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 0)
            for i in range(num_blocks):
                block = []
                for j in range(thickness):
                    # block.append(resize_mask(mask[..., i*thickness + j], target_shape))
                    block.append(resize_mask(mask[..., i + j], target_shape))
                gt_masks_resized.append(block)
            
    return np.array(gt_masks_resized)


def batch_visualization(input_batch, file_name='vis'):
    """input_batch.shape = (bsz, D, H, W)"""
    tar_path = './vis'
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)

    # input_batch = input_batch[np.newaxis, np.newaxis, :]

    sample_count = 0
    for sample in input_batch:
        # if sample_count < 3:
        # slice by slice
        for i in range(len(sample)):
            slc = sample[i]
            # slc = np.transpose(slc, (1, 2, 0))
            # plotting
            slc = np.float32(cv2.resize(slc, (224, 224)))
            plt.imshow(slc, cmap='gray')
            plt.axis('off')
            plt.savefig(
                os.path.join(tar_path, f'{file_name}_{sample_count}_{i}.png'), bbox_inches='tight', pad_inches=0)
            
        sample_count += 1


def batch_visualization_binary(input_batch, file_name='vis'):
    """input_batch.shape = (bsz, D, H, W)"""
    tar_path = './vis'
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)

    # input_batch = input_batch[np.newaxis, np.newaxis, :]

    sample_count = 0
    for sample in input_batch:
        # if sample_count < 3:
        # slice by slice
        for i in range(len(sample)):
            slc = sample[i]
            # slc = np.transpose(slc, (1, 2, 0))
            # plotting
            slc = np.float32(cv2.resize(slc, (224, 224), interpolation=cv2.INTER_NEAREST))
            plt.imshow(slc, cmap='gray')
            plt.axis('off')
            plt.savefig(
                os.path.join(tar_path, f'{file_name}_{sample_count}_{i}.png'), bbox_inches='tight', pad_inches=0)
            
        sample_count += 1


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_iou(x, y):
    intersection = np.bitwise_and(x, y)
    union = np.bitwise_or(x, y)

    iou = 0
    # avoid dividing zero
    if np.sum(union) != 0:
        iou = np.sum(intersection) / np.sum(union)

    return iou


def compute_exp_score(x, y):
    N =  np.sum(y!=0)
    epsilon = 1e-6
    tp = np.sum( x * (y>0))
    tn = np.sum((1-x) * (y<0))
    fp = np.sum( x * (y<0))
    fn = np.sum((1-x) * (y>0))

    exp_precision = tp / (tp + fp + epsilon)
    exp_recall = tp / (tp + fn + epsilon)
    exp_f1 = 2 * (exp_precision * exp_recall) / (exp_precision + exp_recall + epsilon)

    return exp_precision, exp_recall, exp_f1


def calculate_pr_auc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    return pr_auc


def normalize_volume(volume):
    """Normalize the volume"""
    v_min = volume.min(axis=(0, 1), keepdims=True)
    v_max = volume.max(axis=(0, 1), keepdims=True)
    volume = (volume - v_min) / (v_max - v_min + 1e-6)

    return volume


def resize_volume(volume, target_shape):
    """Resize the volume to target_shape"""
    volume_resized = F.interpolate(
        volume,
        size=target_shape,
        mode="trilinear",
        align_corners=True
    )

    return volume_resized


def preprocess_volume(volume, target_shape):
    """Preprocess the volume"""
    volume = normalize_volume(volume)
    # volume = resize_volume(volume, target_shape)
    # volume = centercrop_volume(volume)
    volume = torch.from_numpy(volume)
    volume = torch.unsqueeze(volume, 0)
    volume = torch.unsqueeze(volume, 0)

    volume_resized = resize_volume(volume, target_shape)

    return volume_resized.numpy()


def get_vol_depth(vol):
    """return volume depth"""
    return len(vol[0][0])


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


def record_results(model_name, results):
    csv_file = model_name + '.csv'

    # Open the CSV file in write mode, create it if it doesn't exist
    with open(csv_file, 'a', newline='') as file:
        # Create a CSV writer object
        csv_writer = csv.writer(file)
        csv_writer.writerow(results)


if __name__ == '__main__':
    pass