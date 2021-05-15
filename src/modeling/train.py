import sys, os
import time
import json
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as utils
from src.utils.models import Seldnet_vanilla, Seldnet_augmented
from src.utils.utility_functions import load_model, save_model

'''
Train our baseline model for the Task2 of the L3DAS21 challenge.
This script saves the best model checkpoint, as well as a dict containing
the results (loss and history). To evaluate the performance of the trained model
according to the challenge metrics, please use evaluate_baseline_task2.py.
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
'''

def evaluate(model, device, criterion_sed, criterion_doa, dataloader, batch_size, output_classes, class_overlaps, sed_loss_weight, doa_loss_weight):
    #compute loss without backprop
    model.eval()
    test_loss = 0.
    with tqdm(total=len(dataloader) // batch_size) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            target = target.to(device)
            x = x.to(device)
            t = time.time()
            # Compute loss for each instrument/model
            sed, doa = model(x)
            loss = seld_loss(x, target, model, criterion_sed, criterion_doa, output_classes, class_overlaps, sed_loss_weight, doa_loss_weight)
            test_loss += (1. / float(example_num + 1)) * (loss - test_loss)
            pbar.set_description("Current loss: {:.4f}".format(test_loss))
            pbar.update(1)
    return test_loss


def seld_loss(x, target, model, criterion_sed, criterion_doa, output_classes, class_overlaps, sed_loss_weight, doa_loss_weight):
    '''
    compute seld loss as weighted sum of sed (BCE) and doa (MSE) losses
    '''
    #divide labels into sed and doa  (which are joint from the preprocessing)
    target_sed = target[:,:,:output_classes*class_overlaps]
    target_doa = target[:,:,output_classes*class_overlaps:]

    #compute loss
    sed, doa = model(x)
    print(f'sed model: {sed.shape}')
    print(f'doa model: {doa.shape}')
    print(f'sed target: {target_sed.shape}')
    print(f'doa target: {target_doa.shape}')
    sed = torch.flatten(sed, start_dim=1)
    doa = torch.flatten(doa, start_dim=1)
    target_sed = torch.flatten(target_sed, start_dim=1)
    target_doa = torch.flatten(target_doa, start_dim=1)
    
    print(f'sed model after flatten: {sed.shape}')
    print(f'doa model after flatten: {doa.shape}')
    loss_sed = criterion_sed(sed, target_sed) * sed_loss_weight
    loss_doa = criterion_doa(doa, target_doa) * doa_loss_weight

    return loss_sed + loss_doa


def main(results_path,
         checkpoint_dir,
         training_predictors_path,
         training_target_path,
         validation_predictors_path,
         validation_target_path,
         test_predictors_path,
         test_target_path,
         load_model = None,
         gpu_id = 0,
         use_cuda = False,
         early_stopping = True,
         fixed_seed = False,
         lr = 0.00001,
         batch_size = 3,
         sr = 32000,
         patience = 100,
         architecture = 'seldnet_augmented',
         input_channels = 4,
         class_overlaps = 3,
         time_dim = 4800,
         freq_dim = 256,
         output_classes = 14,
         pool_size = [[8,2],[8,2],[2,2],[1,1]],
         cnn_filters = [64,128,256,512],
         pool_time = True,
         rnn_size = 256,
         n_rnn = 3,
         fc_size = 1024,
         dropout_perc = 0.3,
         n_cnn_filters = 64,
         verbose = False,
         sed_loss_weight = 1.,
         doa_loss_weight = 5.,):

    if use_cuda:
        device = 'cuda:' + str(gpu_id)
    else:
        device = 'cpu'

    if fixed_seed:
        seed = 1
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    #LOAD DATASET
    print ('\nLoading dataset')
    with open(training_predictors_path, 'rb') as f:
        training_predictors = pickle.load(f)
    with open(training_target_path, 'rb') as f:
        training_target = pickle.load(f)
    with open(validation_predictors_path, 'rb') as f:
        validation_predictors = pickle.load(f)
    with open(validation_target_path, 'rb') as f:
        validation_target = pickle.load(f)
    with open(test_predictors_path, 'rb') as f:
        test_predictors = pickle.load(f)
    with open(test_target_path, 'rb') as f:
        test_target = pickle.load(f)

    training_predictors = np.array(training_predictors)
    training_target = np.array(training_target)
    validation_predictors = np.array(validation_predictors)
    validation_target = np.array(validation_target)
    test_predictors = np.array(test_predictors)
    test_target = np.array(test_target)

    print ('\nShapes:')
    print ('Training predictors: ', training_predictors.shape)
    print ('Validation predictors: ', validation_predictors.shape)
    print ('Test predictors: ', test_predictors.shape)
    print ('Training target: ', training_target.shape)
    print ('Validation target: ', validation_target.shape)
    print ('Test target: ', test_target.shape)

    features_dim = int(test_target.shape[-2] * test_target.shape[-1])

    #convert to tensor
    training_predictors = torch.tensor(training_predictors).float()
    validation_predictors = torch.tensor(validation_predictors).float()
    test_predictors = torch.tensor(test_predictors).float()
    training_target = torch.tensor(training_target).float()
    validation_target = torch.tensor(validation_target).float()
    test_target = torch.tensor(test_target).float()
    #build dataset from tensors
    tr_dataset = utils.TensorDataset(training_predictors, training_target)
    val_dataset = utils.TensorDataset(validation_predictors, validation_target)
    test_dataset = utils.TensorDataset(test_predictors, test_target)
    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, batch_size, shuffle=True, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True)

    #LOAD MODEL
    if architecture == 'seldnet_vanilla':
        n_time_frames = test_predictors.shape[-1]
        model = Seldnet_vanilla(time_dim=n_time_frames, freq_dim=freq_dim, input_channels=input_channels,
                    output_classes=output_classes, pool_size=pool_size,
                    pool_time=pool_time, rnn_size=rnn_size, n_rnn=n_rnn,
                    fc_size=fc_size, dropout_perc=dropout_perc,
                    n_cnn_filters=n_cnn_filters, class_overlaps=class_overlaps,
                    verbose=verbose)
    if architecture == 'seldnet_augmented':
        n_time_frames = test_predictors.shape[-1]
        model = Seldnet_augmented(time_dim=n_time_frames, freq_dim=freq_dim, input_channels=input_channels,
                    output_classes=output_classes, pool_size=pool_size,
                    pool_time=pool_time, rnn_size=rnn_size, n_rnn=n_rnn,
                    fc_size=fc_size, dropout_perc=dropout_perc,
                    cnn_filters=cnn_filters, class_overlaps=class_overlaps,
                    verbose=verbose)
        

    if use_cuda:
        print("Moving model to gpu")
    model = model.to(device)

    #compute number of parameters
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))

    #set up the loss functions
    criterion_sed = nn.BCELoss()
    criterion_doa = nn.MSELoss()

    #set up optimizer
    optimizer = Adam(params=model.parameters(), lr=lr)

    #set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf}

    #load model checkpoint if desired
    if load_model is not None:
        print("Continuing training full model from checkpoint " + str(load_model))
        state = load_model(model, optimizer, load_model, use_cuda)

    #TRAIN MODEL
    print('TRAINING START')
    train_loss_hist = []
    val_loss_hist = []
    epoch = 1
    while state["worse_epochs"] < patience:
        print("Training epoch " + str(epoch))
        avg_time = 0.
        model.train()
        train_loss = 0.
        with tqdm(total=len(tr_dataset) // batch_size) as pbar:
            for example_num, (x, target) in enumerate(tr_data):
                target = target.to(device)
                x = x.to(device)
                t = time.time()
                # Compute loss for each instrument/model
                optimizer.zero_grad()
                sed, doa = model(x)
                
                loss = seld_loss(x, target, model, criterion_sed, criterion_doa, output_classes, class_overlaps, sed_loss_weight, doa_loss_weight)
                loss.backward()

                train_loss += (1. / float(example_num + 1)) * (loss - train_loss)
                optimizer.step()
                state["step"] += 1
                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                pbar.update(1)

            #PASS VALIDATION DATA
            val_loss = evaluate(model, device, criterion_sed, criterion_doa, val_data, batch_size, output_classes, class_overlaps, sed_loss_weight, doa_loss_weight)
            print("VALIDATION FINISHED: LOSS: " + str(val_loss))

            # EARLY STOPPING CHECK
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

            if val_loss >= state["best_loss"]:
                state["worse_epochs"] += 1
            else:
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_loss"] = val_loss
                state["best_checkpoint"] = checkpoint_path

                # CHECKPOINT
                print("Saving model...")
                save_model(model, optimizer, state, checkpoint_path)

            state["epochs"] += 1
            #state["worse_epochs"] = 200
            train_loss_hist.append(train_loss.cpu().detach().numpy())
            val_loss_hist.append(val_loss.cpu().detach().numpy())
            epoch += 1

    #LOAD BEST MODEL AND COMPUTE LOSS FOR ALL SETS
    print("TESTING")
    # Load best model based on validation loss
    state = load_model(model, None, state["best_checkpoint"], use_cuda)
    #compute loss on all set_output_size
    train_loss = evaluate(model, device, criterion_sed, criterion_doa, tr_data, batch_size, output_classes, class_overlaps, sed_loss_weight, doa_loss_weight)
    val_loss = evaluate(model, device, criterion_sed, criterion_doa, val_data, batch_size, output_classes, class_overlaps, sed_loss_weight, doa_loss_weight)
    test_loss = evaluate(model, device, criterion_sed, criterion_doa, test_data, batch_size, output_classes, class_overlaps, sed_loss_weight, doa_loss_weight)

    #PRINT AND SAVE RESULTS
    results = {'train_loss': train_loss.cpu().detach().numpy(),
               'val_loss': val_loss.cpu().detach().numpy(),
               'test_loss': test_loss.cpu().detach().numpy(),
               'train_loss_hist': train_loss_hist,
               'val_loss_hist': val_loss_hist}

    print ('RESULTS')
    for i in results:
        if 'hist' not in i:
            print (i, results[i])
    out_path = os.path.join(results_path, 'results_dict.json')
    np.save(out_path, results)



