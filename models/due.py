from utils.resnet import *
from utils.es_utils import *


def inference(sample, to_list, tp_list, predictor):
    
    full_frame = sample.to('cuda:0')
    p = full_frame[:, to_list.to(torch.long), ...]
    
    predictor.predictor.reset_pos_coor(to_list, tp_list)
    predictor = predictor.eval()
    with torch.no_grad():
        rec_past_frames, rec_future_frames, pred = predictor(p.clone())
        
    return pred


def get_weights(var):
    weights = torch.sigmoid(-var)

    weights = weights - torch.min(weights)
    weights = weights / (torch.max(weights) + 1e-6)

    return weights


def train_due(model, predictor, train_loader, val_loader, MODELNAME, num_epochs, task_criterion, attention_criterion, attention_weight, optimizer, use_cuda=True):
    # must use cuda
    device = torch.device('cuda')
    model.to(device)

    # load grad_cam module (target layer = '1' for ResNet18)
    grad_cam = GradCam(model=model, feature_module=model.layer4,
                       target_layer_names=['1'], use_cuda=use_cuda)

    # record best val AUC
    best_val = 0

    # start each epoch
    for epoch in range(num_epochs):

        # Training

        # switch to train mode
        model.train()

        # start timer
        st_train = time.time()

        # print num of epoch
        print(f'Starting epoch {epoch}')

        # records of training
        train_losses = []
        y_pred = []
        y_true = []
        IoU_train = AverageMeter()

        # count num
        pos_count_train = 0

        # start training batches
        for batch_idx, (samples, labels, masks, org_depths, interpolations) in enumerate(train_loader):
            # to cuda
            samples, labels = samples.to(device), labels.to(device)

            # get model prediction
            outputs = model(samples)
            prediction = torch.max(outputs.data, 1)[1]

            # compute prediction loss
            prediction_loss = task_criterion(outputs, labels)
            prediction_loss = torch.mean(prediction_loss)

            # start Explanation Supervision
            model_generated_mask = []
            ground_truth_mask = []
            batch_weights_mask = []
            interpolated_masks = []

            # get model generated masks and ground truth masks
            for sample, label, mask, original_depth, interp in zip(samples, labels, masks, org_depths, interpolations):
    
                # filter masks, leave *only* the pos
                if torch.max(mask) > 0:
                    att_map = grad_cam.get_attention_map(torch.unsqueeze(sample, 0), index=label)
                    # (1, C, D, H, W) -> (C, D, H, W)
                    att_map = torch.squeeze(att_map, 0)
                    # (C, D, H, W) -> (D, H, W)
                    att_map = torch.squeeze(att_map, 0)
                    # (C, D, H, W) -> (D, H, W)
                    mask = torch.squeeze(mask, 0)
                    # (C, D, H, W) -> (D, H, W)
                    interp = torch.squeeze(interp, 0)

                    # get annotation weights
                    C = 1
                    D = mask.shape[0]
                    H = mask.shape[1]
                    W = mask.shape[2]

                    slice_start = 0
                    slice_stop = D - 1
                    
                    annotated_slices = np.linspace(slice_start, slice_stop, original_depth, endpoint=True)
                    annotated_slices = list(map(round, annotated_slices))
                
                    mask_weighted = torch.ones(size=(D, H, W), dtype=torch.float32)
                    
                    block_num = len(annotated_slices) - 1
                    
                    for b_id in range(block_num):
                        block = []

                        start_loc = annotated_slices[b_id]
                        end_loc = annotated_slices[b_id + 1]
                        gap_distance = end_loc - start_loc - 1

                        if gap_distance < 1:
                            continue

                        gap_locations = list(np.arange(start_loc + 1, end_loc))

                        context_p = mask[start_loc]
                        context_f = mask[end_loc]

                        block.append(context_p)
                        block.append(context_f)
                        block = np.array(block).reshape(1, 2, C, H, W)
                        block = torch.from_numpy(block).float()

                        predictor = predictor.eval()
                        with torch.no_grad():

                            to_list = torch.from_numpy(np.array([0, 1])).to(torch.float32)
                            tp_list = torch.from_numpy(np.linspace(2, 6, gap_distance, endpoint=True)).to(torch.float32)
                            vrc_hfps = inference(block, to_list, tp_list, predictor)
        
                        vrc_hfps = vrc_hfps.reshape(gap_distance, H, W)

                        mask_weights = get_weights(vrc_hfps)
                        mask_weighted[gap_locations] = mask_weights.cpu()

                    mask_weighted = mask_weighted.to(device)
                    mask = mask.to(device)
                    interp = interp.to(device)
                    
                    model_generated_mask.append(att_map)
                    ground_truth_mask.append(mask)
                    batch_weights_mask.append(mask_weighted)
                    interpolated_masks.append(interp)

                    pos_count_train += 1
            
            attention_loss = 0

            # if any pos exists in this batch 
            if model_generated_mask:

                model_generated_mask = torch.stack(model_generated_mask)
                ground_truth_mask = torch.stack(ground_truth_mask)
                batch_weights_mask = torch.stack(batch_weights_mask)
                interpolated_masks = torch.stack(interpolated_masks)

                # binary loss
                temp1 = torch.tanh(5 * (model_generated_mask - 0.5))
                temp_loss = attention_criterion(temp1, ground_truth_mask)
                temp_loss = temp_loss * batch_weights_mask
                # normalize by effective areas
                temp_size = (ground_truth_mask != 0).float()
                eff_loss = torch.sum(temp_loss * temp_size) / torch.sum(temp_size)
                attention_loss += torch.relu(torch.mean(eff_loss))

                # continuous loss
                tempD = attention_criterion(model_generated_mask, interpolated_masks)
                attention_loss += torch.mean(batch_weights_mask * tempD)

            loss = prediction_loss + attention_weight * attention_loss
            
            print('Batch:', batch_idx, 'Total loss:', loss, 'Prediction loss:', prediction_loss,
                      'Attention loss:', attention_weight * attention_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            y_pred += prediction.cpu().detach().tolist()
            y_true += labels.cpu().detach().tolist()
        
        # end timer
        et_train = time.time()
        time_train = et_train - st_train

        # compute metrics
        Acc_train = accuracy_score(y_true, y_pred)
        # summarize the epoch
        print('Epoch:', epoch, 'Train Time:', time_train, 'Train Loss:', np.average(train_losses), 'Acc_train:', Acc_train)
        
        best_val = model_val(model, val_loader, MODELNAME, best_val)


def model_val(model, val_loader, MODELNAME, best_val):
    # must use cuda
    device = torch.device('cuda')
    model.to(device)
    # switch to eval mode
    model.eval()

    # load grad_cam module (target layer = '1' for ResNet18)
    grad_cam = GradCam(model=model, feature_module=model.layer4,
                            target_layer_names=['1'], use_cuda=True)
    
    # start timer
    st_val = time.time()

    # records of validation
    y_pred_val = []
    y_true_val = []
    y_pred_val_softmax = []
    IoU_val = AverageMeter()

    # count num
    pos_count = 0
    count_for_vis_val = 0

    # start validation batches
    for batch_idx, (samples, labels, masks) in enumerate(val_loader):
        # to cuda
        samples, labels = samples.to(device), labels.to(device)

        # get model prediction
        with torch.no_grad():
            outputs = model(samples)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.max(outputs.data, 1)[1]
        # append to lists
        y_pred_val_softmax += outputs.cpu().detach().tolist()
        y_pred_val += prediction.cpu().detach().tolist()
        y_true_val += labels.cpu().detach().tolist()

        # compute IoU one by one
        for sample, label, mask in zip(samples, labels, masks):

            # filter masks, leave *only* the pos
            if torch.max(mask) > 0:

                # get ground truth mask
                # (C, D, H, W) -> (D, H, W)
                target_att = torch.squeeze(mask, 0)
                # to numpy
                target_att = target_att.cpu().detach().numpy()
                # bianarize ground truth mask (True/False -> 1/0)
                target_att_binary = (target_att > 0.5)

                # get model generated mask for computing IoU
                model_generated_mask_for_vis = grad_cam(
                    torch.unsqueeze(sample, 0))

                # (1, C, D, H, W) -> (C, D, H, W)
                model_generated_mask_for_vis = torch.squeeze(
                    model_generated_mask_for_vis, 0)
                # (C, D, H, W) -> (D, H, W)
                model_generated_mask_for_vis = torch.squeeze(
                    model_generated_mask_for_vis, 0)
                # to numpy
                model_generated_mask_for_vis = model_generated_mask_for_vis.cpu().detach().numpy()

                # bianarize model generated mask
                item_att_binary = (model_generated_mask_for_vis > 0.5)

                # compute IoU (use binary masks)
                IoU = compute_iou(item_att_binary, target_att_binary)
                # add to IoU_val
                IoU_val.update(IoU, 1)
                # # check IoU
                # print('Num', pos_count, 'IoU:', IoU)

                pos_count += 1

    # end timer
    et_val = time.time()
    time_val = et_val - st_val

    curr_val = calculate_pr_auc(y_true_val, np.array(y_pred_val_softmax)[:, 1])
    IoU_val_avg = IoU_val.avg

    # save checkpoint model
    if curr_val > best_val:
        # update best_val metric
        best_val = curr_val
        # alert
        print('Update!!! Best Model in val:', best_val)

        tar_path = './checkpoints/'
        if not os.path.exists(tar_path):
            os.makedirs(tar_path)
        torch.save(model, os.path.join(tar_path, MODELNAME))

    print('Val Time:', time_val, 'curr_val:', curr_val, 'best_val:', best_val, 'IoU_val_avg:', IoU_val_avg)

    return best_val


def model_test(model, test_loader, output_heatmap=False, use_cuda=True):
    # must use cuda
    device = torch.device('cuda')
    model.to(device)

    # switch to eval mode
    model.eval()

    # load grad_cam module (target layer = '1' for ResNet18)
    grad_cam_test = GradCam(model=model, feature_module=model.layer4,
                            target_layer_names=['1'], use_cuda=use_cuda)

    # start timer
    st_test = time.time()

    # records of testing
    y_pred_test = []
    y_true_test = []
    y_pred_test_softmax = []
    IoU_test = AverageMeter()
    exp_precision = AverageMeter()
    exp_recall = AverageMeter()
    exp_f1 = AverageMeter()

    # count num
    pos_count_test = 0
    count_for_vis_test = 0

    # start testing batches
    for batch_idx, (samples, labels, masks) in enumerate(test_loader):
        # to cuda
        samples, labels = samples.to(device), labels.to(device)

        # get model prediction
        with torch.no_grad():
            outputs = model(samples)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.max(outputs.data, 1)[1]
        # append to lists
        y_pred_test_softmax += outputs.cpu().detach().tolist()
        y_pred_test += prediction.cpu().detach().tolist()
        y_true_test += labels.cpu().detach().tolist()

        # compute IoU one by one
        for sample, label, mask in zip(samples, labels, masks):

            # filter masks, leave *only* the pos
            if torch.max(mask) > 0:

                # get ground truth mask
                # (C, D, H, W) -> (D, H, W)
                target_att = torch.squeeze(mask, 0)
                # to numpy
                target_att = target_att.cpu().detach().numpy()
                # bianarize ground truth mask (True/False -> 1/0)
                target_att_binary = (target_att > 0.5)

                # get model generated mask for computing IoU
                model_generated_mask_for_vis = grad_cam_test(
                    torch.unsqueeze(sample, 0))

                # (1, C, D, H, W) -> (C, D, H, W)
                model_generated_mask_for_vis = torch.squeeze(
                    model_generated_mask_for_vis, 0)
                # (C, D, H, W) -> (D, H, W)
                model_generated_mask_for_vis = torch.squeeze(
                    model_generated_mask_for_vis, 0)
                # to numpy
                model_generated_mask_for_vis = model_generated_mask_for_vis.cpu().detach().numpy()

                # bianarize model generated mask
                item_att_binary = (model_generated_mask_for_vis > 0.5)

                # compute IoU (use binary masks)
                IoU = compute_iou(item_att_binary, target_att_binary)
                # add to IoU_val
                IoU_test.update(IoU, 1)
                
                p, r, f1 = compute_exp_score(item_att_binary, target_att)
                exp_precision.update(p.item(), 1)
                exp_recall.update(r.item(), 1)
                exp_f1.update(f1.item(), 1)

                pos_count_test += 1

    # end timer
    et_test = time.time()
    time_test = et_test - st_test

    # compute metrics
    acc_test = accuracy_score(y_true_test, y_pred_test)
    AUC_test = roc_auc_score(y_true_test, np.array(y_pred_test_softmax)[:, 1])
    PR_AUC_test = calculate_pr_auc(y_true_test, np.array(y_pred_test_softmax)[:, 1])

    precision_test = precision_score(y_true_test, y_pred_test)
    recall_test = recall_score(y_true_test, y_pred_test)
    F1_test = f1_score(y_true_test, y_pred_test)

    TN, FP, FN, TP = confusion_matrix(y_true_test, y_pred_test).ravel()
    specificity_test = TN / float(TN + FP)
    IoU_test_avg = IoU_test.avg

    # summarize testing
    print('Finish Testing on Test Set. Time:', time_test, 'Acc:', acc_test, 'AUC:', AUC_test, 'PR-AUC:', PR_AUC_test, 'Precision:', precision_test,
          'Recall:', recall_test, 'F1:', F1_test, 'Specificity:', specificity_test, 'IoU:', IoU_test_avg, exp_precision.avg, exp_recall.avg, exp_f1.avg)
    
    return (acc_test, AUC_test, PR_AUC_test, precision_test, recall_test, F1_test, specificity_test, IoU_test_avg, exp_precision.avg, exp_recall.avg, exp_f1.avg)


if __name__ == '__main__':
    pass
