import os
import time
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from hydra import compose, initialize

from ... import LitPredictor
from utils import get_bal_lidc_dataset, generate_model, record_results, get_lidc_train_due, \
    get_lidc_dataset_imbal, get_bal_pancreas_dataset, get_pancreas_dataset_imbal, get_pancreas_train_due
from models import model_test, train_due


def main():
    parser = argparse.ArgumentParser(description='DUE')
    parser.add_argument('--model', type=str, default='due',
                        help='Model name.')
    parser.add_argument('--dataset', type=str, default='Pancreas',
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--attention_weight', type=float, default=1.0, 
                        help='Scale factor for explanation loss')
    parser.add_argument('--seed', type=int, default=0,
                        help='The random seed.')
    parser.add_argument('--cuda', type=str, default=True,
                        help='Use GPU or not')
    parser.add_argument('--n_epoch', type=int, default=50,
                        help='Number of epoch to run')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='training batchsize (default: 16)')
    
    args = parser.parse_args()
    
    AW = args.attention_weight
    BATCH_SIZE = args.batch_size
    EPOCH = args.n_epoch
    SEED = args.seed

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    num_classes = 2
    # size of input data
    C, H, W, D = 1, 128, 128, 64
    
    if args.model == 'baseline':
        model_name = f'{args.model}_seed{args.seed}.pt'
    else:
        model_name = f'{args.model}_aw{args.attention_weight}_seed{args.seed}.pt'

    if args.dataset == 'Pancreas':
        model_name = 'pancreas_' + model_name
     
    # Prepare data
    st_data = time.time()
    
    if args.dataset == 'LIDC':

        if args.model == 'due' or args.model == 'all':
            train_set = get_lidc_train_due('train', C, H, W, D)
        else:
            train_set = get_bal_lidc_dataset('train', C, H, W, D)

        val_set = get_lidc_dataset_imbal('val', C, H, W, D)
        test_set = get_lidc_dataset_imbal('test', C, H, W, D)

    elif args.dataset == 'Pancreas':

        if args.model == 'due' or args.model == 'all':
            train_set = get_pancreas_train_due('train', C, H, W, D)
        else:
            train_set = get_bal_pancreas_dataset('train', C, H, W, D)

        val_set = get_pancreas_dataset_imbal('val', C, H, W, D)
        test_set = get_pancreas_dataset_imbal('test', C, H, W, D)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False)

    et_data = time.time()
    time_data = et_data - st_data
    print('Data preparation finished, spent', time_data, 's.',
          'Train:', len(train_set), 'Val:', len(val_set), 'Test:', len(test_set))

    # Setup model
    print(model_name, args)
    backbone = generate_model(18, n_input_channels=1, n_classes=num_classes)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=0.001)

    task_criterion = nn.CrossEntropyLoss(reduction='none')
    attention_criterion = nn.L1Loss(reduction='none')

    if args.model == 'due':
        initialize(config_path="./your/path/to/configs")
        cfg = compose(config_name="your_config_name")
        cfg.Env.strategy = 'dp'
        cfg.Predictor.resume_AE_ckpt = '/your/path/to/checkpoint.ckpt'
        vae = LitPredictor.load_from_checkpoint('/your/path/to/checkpoint.ckpt', cfg=cfg).to('cuda:0')
        
        train_due(backbone, vae, train_loader, val_loader, model_name, EPOCH, task_criterion, attention_criterion, AW, optimizer)

    test_model = torch.load(os.path.join('/your/path/checkpoints/', model_name))
    results = model_test(test_model, test_loader)
    results = results + (args,)
    record_results('results', results)


if __name__ == '__main__':
    main()
