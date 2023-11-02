#!/usr/bin/python


from split_cv import transform_Data
from split_Few_shot import transform_Data_FewShot
from Analysis import  new_folder, Rep_histograms, Models_trained, Rep_heatmap

import argparse
import os
import numpy as np
import torch
import random
import main_diffpool
import main_gat
import main_gcn
import main_gunet
import main_sag
import time

def train_main_model(dataset,model,view, cv_number, cv_current):
    
    """
    Parameters
    ----------
    dataset : dataset
    model : GNN model (diffpool, gat, gcn, gunet or sag)
    view : index of cortical morphological network. 
    
    Description
    ----------
    This method trains selected GNN model with 5-Fold Cross Validation.
    
    """
    name = str(cv_number)+"Fold"
    #name = "5Fold"
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)    
    
    model_name = "MainModel_"+name+"_"+dataset+ "_" + model + "_view_" + str(view) + "_CV_" + str(cv_current) 
    new_folder(model)
    if model=='diffpool':
        main_diffpool.test_scores(dataset, view, model_name, cv_number, cv_current, model)
    elif model=='gcn':
        main_gcn.test_scores(dataset, view, model_name, cv_number, cv_current, model)
    elif model=='gat':
        main_gat.test_scores(dataset, view, model_name, cv_number, cv_current, model)
    elif model == "gunet":
        transform_Data(cv_number, dataset)
        main_gunet.cv_benchmark(dataset, view, cv_number, cv_current, model)
    elif model == "sag":
        transform_Data(cv_number, dataset)
        main_sag.cv_benchmark(dataset, view, cv_number, cv_current, model, model_name)

def calculate_kappa(dataset,model,view, cv_number, cv_current, Model):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)    

    name = str(cv_number)+"Fold"
    model_name = Model+"_MainModel_"+name+"_"+dataset+ "_" + model + "_view_" + str(view) + "_CV_" + str(cv_current) 

    if model=='diffpool':
        main_diffpool.test_kappa(dataset, view, model_name, cv_number, cv_current, model)
    elif model=='gcn':
        main_gcn.test_kappa(dataset, view, model_name, cv_number, cv_current, model)
    elif model=='gat':
        #main_gat.test_scores(dataset, view, model_name, cv_number, cv_current, model)
        print('GAT')
        pass
    elif model == "gunet":
        #transform_Data(cv_number, dataset)
        #main_gunet.cv_benchmark(dataset, view, cv_number, cv_current, model)
        print('Gunet')
        pass
    elif model == "sag":
        #transform_Data(cv_number, dataset)
        #main_sag.cv_benchmark(dataset, view, cv_number, cv_current, model, model_name)    
        print('SAG')
        pass

def two_shot_train(dataset, model, view, num_shots):
    """
    Parameters
    ----------
    dataset : dataset
    model : GNN model (diffpool, gat, gcn, gunet or sag)
    view : index of cortical morphological network.
    
    Description
    ----------
    This method trains selected GNN model with Two shot learning.
    
    """
    
    #torch.manual_seed(0)
    #np.random.seed(0)
    #random.seed(0)
    transform_Data_FewShot(dataset)
    new_folder(model)
    if model == "gunet":
        main_gunet.two_shot_trainer(dataset, view, num_shots)
    elif model == "gcn":
        main_gcn.two_shot_trainer(dataset, view, num_shots)
    elif model == "gat":
        main_gat.two_shot_trainer(dataset, view, num_shots)
    elif model == "diffpool":
        main_diffpool.two_shot_trainer(dataset, view, num_shots)
    elif model == "sag":
        main_sag.two_shot_trainer(dataset, view, num_shots)

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'kappa'])
    #parser.add_argument('--v', type=str, default=0, help='index of cortical morphological network.')
    parser.add_argument('--cv_number', type=int, default=3, help='number of cross validations.')
    parser.add_argument('--cv_current', type=int, default=0, help='number of cross validations.')
    parser.add_argument('--num_shots', type=str, default=5, help='number of runs for the FS learning.')
    parser.add_argument('--view', type=int, default=0, help='view no to train a model')
    parser.add_argument('--model', type=str, default='none', help='GNN architecture to train')
    parser.add_argument('--dataset', type=str, default='none', help='the dataset to train a model')
    
    #parser.add_argument('--data', type=str, default='Demo', choices = [ f.path[5:] for f in os.scandir("data") if f.is_dir() ], help='Name of dataset')
    args = parser.parse_args()
    #view = args.v
    #dataset = args.data
    num_shots = args.num_shots
    cv_number = args.cv_number
    cv_current = args.cv_current
    view_i = args.view
    model = args.model
    dataset_i = args.dataset
    Model = 'Diffpool'

    if args.mode == 'train':

        '''
        Training GNN Models with datasets of data directory.
        '''
        #datasets_asdnc = ['Demo']
        #datasets_asdnc = ['RH_ASDNC'] # 'RH_ASDNC_d2', 'RH_ASDNC_d3', '
        #datasets_adlmci = ['RH_ADLMCI','LH_ADLMCI']
        print(f"{dataset_i} {model} {view_i}  {cv_current} / {cv_number} ")
        #time.sleep(2)
        #two_shot_train(dataset_i, model, view_i, num_shots)
        train_main_model(dataset_i, model, view_i, cv_number, cv_current)
    
        print("All GNN architectures are trained with dataset: "+dataset_i)
          
        
    elif args.mode == 'kappa':
        calculate_kappa(dataset_i, model, view_i, cv_number, cv_current, Model)
