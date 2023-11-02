# -*- coding: utf-8 -*-
from sklearn import preprocessing
from torch.autograd import Variable
from models_gcn import GCN

import os
import torch
import numpy as np
import argparse
import pickle
import sklearn.metrics as metrics
import cross_val
import time
import random


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate(dataset, model_GCN, args, threshold_value, model_name):
    """
    Parameters
    ----------
    dataset : dataloader (dataloader for the validation/test dataset).
    model_GCN : nn model (GCN model).
    args : arguments
    threshold_value : float (threshold for adjacency matrices).
    
    Description
    ----------
    This methods performs the evaluation of the model on test/validation dataset
    
    Returns
    -------
    test accuracy.
    """
    model_GCN.eval()
    labels = []
    preds = []
    avg_loss = 0

    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
        
        adj = torch.squeeze(adj)
        
        features = np.identity(adj.shape[0])
        
        features = Variable(torch.from_numpy(features).float(), requires_grad=False).cpu()
        if args.threshold in ["median", "mean"]:
            adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))
        
        ypred = model_GCN(features, adj)

        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        label = Variable(data['label'].long()).to(device)
        avg_loss += model_GCN.loss(ypred, label)

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    simple_r = {'labels':labels,'preds':preds}

    result = {'prec': metrics.precision_score(labels, preds, average='binary'),
              'recall': metrics.recall_score(labels, preds, average='binary'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="binary"),
              'specificity': metrics.recall_score(labels, preds, pos_label=0, average='binary'),
              'val loss': float(avg_loss.detach().cpu().numpy()) / len(dataset)} # specificity is the recall of the negative class . You can reach it just setting the pos_label parameter

    if args.evaluation_method == 'model assessment':
        name = 'Test'
    if args.evaluation_method == 'model selection':
        name = 'Validation'
    
    print('VALIDATON SCORES', result)
    
    if result['F1'] >= args.best_f1:
        args.best_f1_changed = True
        args.best_f1 = result['F1']
        print('Best F1 Obtained: ', args.best_f1)

        with open("./gcn/Labels_and_preds/"+model_name+"_test.pickle", 'wb') as f:
            pickle.dump(simple_r, f)
        print('Labels_and_preds updated')
        w_dict = {"w": model_GCN.LinearLayer.weight}
        with open("GCN_W.pickle", 'wb') as f:
            pickle.dump(w_dict, f)
            print('GCN_W weights saved')

    return result

def minmax_sc(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x

def train(args, train_dataset, val_dataset, model_GCN, threshold_value, model_name):
    """
    Parameters
    ----------
    args : arguments
    train_dataset : dataloader (dataloader for the validation/test dataset).
    val_dataset : dataloader (dataloader for the validation/test dataset).
    model_GCN : nn model (GCN model).
    threshold_value : float (threshold for adjacency matrices).
    
    Description
    ----------
    This methods performs the training of the model on train dataset and calls evaluate() method for evaluation.
    
    Returns
    -------
    test accuracy.
    """
    params = list(model_GCN.parameters()) 
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    test_accs = []
    train_loss=[]
    val_acc=[]
    

    for epoch in range(args.num_epochs):
        print("Epoch ",epoch)
        
        #print("Size of Training Set:" + str(len(train_dataset)))
        #print("Size of Validation Set:" + str(len(val_dataset)))
        model_GCN.train()
        total_time = 0
        avg_loss = 0.0
        
        preds = []
        labels = []
        print(len(train_dataset))
        
        for batch_idx, data in enumerate(train_dataset):
            begin_time = time.time()
            
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            #adj_id = Variable(data['id'].int()).to(device)
            
            adj = torch.squeeze(adj)

            features = np.identity(adj.shape[0])
            features = Variable(torch.from_numpy(features).float(), requires_grad=False).cpu()
            if args.threshold in ["median", "mean"]:
                adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))
            
            ypred= model_GCN(features, adj)
            
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())
            loss = model_GCN.loss(ypred, label)
            model_GCN.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model_DIFFPOOL.parameters(), args.clip)
            optimizer.step()
            
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
        
        
        # Save weights of GNN model
        if epoch==args.num_epochs-1:
              model_GCN.is_trained = True
              
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        simple_r = {'labels':labels,'preds':preds}

        
        result = {'prec': metrics.precision_score(labels, preds, average='binary'),
                    'recall': metrics.recall_score(labels, preds, average='binary'),
                    'acc': metrics.accuracy_score(labels, preds),
                    'F1': metrics.f1_score(labels, preds, average="binary"),
                    'specificity': metrics.recall_score(labels, preds, pos_label=0, average='binary'),
                    'train loss': float(avg_loss.detach().cpu().numpy()) / len(train_dataset)} # specificity is the recall of the negative class . You can reach it just setting the pos_label parameter
        print('TRAINING SCORES', result)


        #print("Train accuracy : ", np.mean( preds == labels ))
        test_acc = evaluate(val_dataset, model_GCN, args, threshold_value,model_name)
        val_acc.append(test_acc)
        test_accs.append(test_acc)
        train_loss.append(avg_loss)

        path = './gcn/weights/W_'+model_name+'.pickle'
    
        #if os.path.exists(path):
        #    os.remove(path)

        if args.best_f1_changed:
            os.rename('GCN_W.pickle',path)

            los_p = {'loss':train_loss}
            with open("./gcn/training_loss/Training_loss_"+model_name+".pickle", 'wb') as f: # training_loss
                pickle.dump(los_p, f)
            torch.save(model_GCN,"./gcn/models/GCN_"+model_name+".pt") # models
            
            print('Training loss, model, weights updated')
            args.best_f1_changed = False
    
            with open("./gcn/Labels_and_preds/"+model_name+"_train.pickle", 'wb') as f:
                pickle.dump(simple_r, f)

    args.best_f1_changed = False
    args.best_f1 = 0
    
    return test_acc

def load_data(args):
    """
    Parameters
    ----------
    args : arguments
    Description
    ----------
    This methods loads the adjacency matrices representing the args.view -th view in dataset
    
    Returns
    -------
    List of dictionaries{adj, label, id}
    """
    #Load graphs and labels


    with open('data/'+args.dataset+'/'+args.dataset+'_edges','rb') as f:
        multigraphs = pickle.load(f)        
    with open('data/'+args.dataset+'/'+args.dataset+'_labels','rb') as f:
        labels = pickle.load(f)
    adjacencies = [multigraphs[i][:,:,args.view] for i in range(len(multigraphs))]
    #Normalize inputs
    if args.NormalizeInputGraphs==True:
        for subject in range(len(adjacencies)):
            adjacencies[subject] = minmax_sc(adjacencies[subject])
    
    #Create List of Dictionaries
    G_list=[]
    for i in range(len(labels)):
        G_element = {"adj":   adjacencies[i],"label": labels[i],"id":  i,}
        G_list.append(G_element)
    return G_list

def arg_parse(dataset, view, num_shots=2, cv_number=5, cv_current=0, model='none'):
    """
    arguments definition method
    """
    parser = argparse.ArgumentParser(description='Graph Classification')
    
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--v', type=str, default=1)
    parser.add_argument('--data', type=str, default='Sample_dataset', choices = [ f.path[5:] for f in os.scandir("data") if f.is_dir() ])
        
    parser.add_argument('--dataset', type=str, default=dataset,
                        help='Dataset')
    parser.add_argument('--view', type=int, default=view,
                        help = 'view index in the dataset')
    parser.add_argument('--num_epochs', type=int, default=50, #50
                        help='Training Epochs')
    parser.add_argument('--num_shots', type=int, default=num_shots, #100
                        help='number of shots')
    parser.add_argument('--cv_number', type=int, default=cv_number,
                        help='number of validation folds.')
    parser.add_argument('--cv_current', type=int, default=cv_current,
                        help='number of validation folds.')            
    parser.add_argument('--NormalizeInputGraphs', default=False, action='store_true',
                        help='Normalize Input adjacency matrices of graphs')
    parser.add_argument('--evaluation_method', type=str, default='model assessment',
                        help='evaluation method, possible values : model selection, model assessment')
    ##################
    parser.add_argument('--threshold', dest='threshold', default='mean',
            help='threshold the graph adjacency matrix. Possible values: no_threshold, median, mean')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=2,
                        help='Number of label classes')
    parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
    parser.add_argument('-best_f1', type=int, default=0, help='keeps the best f1 during training')
    parser.add_argument('-best_f1_changed', type=bool, default=False, help='')

    ##################
    parser.add_argument('--model', type=str, default='none', help='GNN architecture to train')

    return parser.parse_args()

def get_fold_number(ll):
    random.seed(random.randrange(100))
    return ll.pop(random.randrange(len(ll)))

def benchmark_task(args, model_name):
    """
    Parameters
    ----------
    args : Arguments
    Description
    ----------
    Initiates the model and performs train/test or train/validation splits and calls train() to execute training and evaluation.
    Returns
    -------
    test_accs : test accuracies (list)

    """
    G_list = load_data(args)
    num_nodes = G_list[0]['adj'].shape[0]
    test_accs = []
    folds = cross_val.stratify_splits(G_list,args)
    [random.shuffle(folds[i]) for i in range(len(folds))]

    #for i in range(args.cv_number):
    print('rand cv: ', args.cv_current)
    train_set, validation_set, test_set = cross_val.datasets_splits(folds, args, args.cv_current)
    #time.sleep(2)
    if args.evaluation_method =='model selection':
        train_dataset, val_dataset, threshold_value = cross_val.model_selection_split(train_set, validation_set, args)
    
    if args.evaluation_method =='model assessment':
        train_dataset, val_dataset, threshold_value = cross_val.model_assessment_split(train_set, validation_set, test_set, args)

    model_GCN = GCN(nfeat = num_nodes,
                    nhid = args.hidden,
                    nclass = args.num_classes,
                    dropout = args.dropout)
    
    test_acc = train(args, train_dataset, val_dataset, model_GCN, threshold_value, model_name+"_view_"+str(args.view)+"_CV_"+str(args.cv_current))
    test_accs.append(test_acc)
    return test_accs

def test_scores(dataset, view, model_name, cv_number, cv_current, model):
    args = arg_parse(dataset, view, cv_number=cv_number, cv_current=cv_current, model=model)
    print("Main : ",args)

    G_list = load_data(args)
    num_nodes = G_list[0]['adj'].shape[0]
    test_accs = []
    folds = cross_val.stratify_splits(G_list,args)
    [random.shuffle(folds[i]) for i in range(len(folds))]

    #for i in range(args.cv_number):
    print('rand cv: ', args.cv_current)
    train_set, validation_set, test_set = cross_val.datasets_splits(folds, args, args.cv_current)
    #time.sleep(2)
    if args.evaluation_method =='model selection':
        train_dataset, val_dataset, threshold_value = cross_val.model_selection_split(train_set, validation_set, args)
    
    if args.evaluation_method =='model assessment':
        train_dataset, val_dataset, threshold_value = cross_val.model_assessment_split(train_set, validation_set, test_set, args)

    model_GCN = GCN(nfeat = num_nodes,
                    nhid = args.hidden,
                    nclass = args.num_classes,
                    dropout = args.dropout)
    
    test_acc = train(args, train_dataset, val_dataset, model_GCN, threshold_value, model_name)
    test_accs.append(test_acc)
    return test_accs

    return test_accs
        
def test_kappa(dataset, view, model_name, cv_number, cv_current, model):
    def get_preds(val_dataset, threshold_value):
        #model_GCN = GCN(nfeat = num_nodes,
        #                nhid = args.hidden,
        #                nclass = args.num_classes,
        #                dropout = args.dropout)
        #model_GCN.load_state_dict(torch.load(f"{model}/models/{model_name}.pt"))
        model_GCN = torch.load(f"{model}/models/{model_name}.pt")
        model_GCN.eval()

        labels = []
        preds = []

        for batch_idx, data in enumerate(val_dataset): # val_dataset_f1
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            labels.append(data['label'].long().numpy())
            
            adj = torch.squeeze(adj)
            
            features = np.identity(adj.shape[0])
            
            features = Variable(torch.from_numpy(features).float(), requires_grad=False).cpu()
            if args.threshold in ["median", "mean"]:
                adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0])) # threshold_value_f1
            
            ypred = model_GCN(features, adj)

            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())

        labels = np.hstack(labels)
        preds = np.hstack(preds)

        simple_r = {'labels':labels,'preds':preds}
        return simple_r


    args = arg_parse(dataset, view, cv_number=cv_number, cv_current=cv_current, model=model)
    print("Main : ",args)
    

    args.view = 0
    G_list_v1 = load_data(args)
    args.view = 1
    G_list_v2 = load_data(args)
    args.view = 2
    G_list_v3 = load_data(args)
    args.view = 3
    G_list_v4 = load_data(args)
    args.view = 4
    G_list_v5 = load_data(args)
    args.view = 5
    G_list_v6 = load_data(args)

    num_nodes = G_list_v1[0]['adj'].shape[0]

    folds_v1 = cross_val.stratify_splits(G_list_v1,args) 
    [random.shuffle(folds_v1[i]) for i in range(len(folds_v1))]

    folds_v2 = cross_val.stratify_splits(G_list_v2,args) 
    [random.shuffle(folds_v2[i]) for i in range(len(folds_v2))]

    folds_v3 = cross_val.stratify_splits(G_list_v3,args) 
    [random.shuffle(folds_v3[i]) for i in range(len(folds_v3))]

    folds_v4 = cross_val.stratify_splits(G_list_v4,args) 
    [random.shuffle(folds_v4[i]) for i in range(len(folds_v4))]

    folds_v5 = cross_val.stratify_splits(G_list_v5,args) 
    [random.shuffle(folds_v5[i]) for i in range(len(folds_v5))]

    folds_v6 = cross_val.stratify_splits(G_list_v6,args) 
    [random.shuffle(folds_v6[i]) for i in range(len(folds_v6))]


    print('rand cv: ', args.cv_current)
    train_set_v1, validation_set_v1, test_set_v1 = cross_val.datasets_splits(folds_v1, args, cv_current)
    train_set_v2, validation_set_v2, test_set_v2 = cross_val.datasets_splits(folds_v2, args, cv_current)
    train_set_v3, validation_set_v3, test_set_v3 = cross_val.datasets_splits(folds_v3, args, cv_current)
    train_set_v4, validation_set_v4, test_set_v4 = cross_val.datasets_splits(folds_v4, args, cv_current)
    train_set_v5, validation_set_v5, test_set_v5 = cross_val.datasets_splits(folds_v5, args, cv_current)
    train_set_v6, validation_set_v6, test_set_v6 = cross_val.datasets_splits(folds_v6, args, cv_current)


    if args.evaluation_method =='model assessment':
        _, val_dataset_v1, threshold_value_v1 = cross_val.model_assessment_split(train_set_v1, validation_set_v1, test_set_v1, args)
        _, val_dataset_v2, threshold_value_v2 = cross_val.model_assessment_split(train_set_v2, validation_set_v2, test_set_v2, args)
        _, val_dataset_v3, threshold_value_v3 = cross_val.model_assessment_split(train_set_v3, validation_set_v3, test_set_v3, args)
        _, val_dataset_v4, threshold_value_v4 = cross_val.model_assessment_split(train_set_v4, validation_set_v4, test_set_v4, args)
        _, val_dataset_v5, threshold_value_v5 = cross_val.model_assessment_split(train_set_v5, validation_set_v5, test_set_v5, args)
        _, val_dataset_v6, threshold_value_v6 = cross_val.model_assessment_split(train_set_v6, validation_set_v6, test_set_v6, args)


    simple_r_v1 = get_preds(val_dataset_v1, threshold_value_v1)
    simple_r_v2 = get_preds(val_dataset_v2, threshold_value_v2)
    simple_r_v3 = get_preds(val_dataset_v3, threshold_value_v3)
    simple_r_v4 = get_preds(val_dataset_v4, threshold_value_v4)
    simple_r_v5 = get_preds(val_dataset_v5, threshold_value_v5)
    simple_r_v6 = get_preds(val_dataset_v6, threshold_value_v6)

    if not os.path.exists(f"{model}/kappa"):
        os.mkdir(f"{model}/kappa")
    
    with open(f"{model}/kappa/{model_name}_test_with_view_0.pickle", 'wb') as f:
      pickle.dump(simple_r_v1, f)
    with open(f"{model}/kappa/{model_name}_test_with_view_1.pickle", 'wb') as f:
      pickle.dump(simple_r_v2, f)
    with open(f"{model}/kappa/{model_name}_test_with_view_2.pickle", 'wb') as f:
      pickle.dump(simple_r_v3, f)
    with open(f"{model}/kappa/{model_name}_test_with_view_3.pickle", 'wb') as f:
      pickle.dump(simple_r_v4, f)
    with open(f"{model}/kappa/{model_name}_test_with_view_4.pickle", 'wb') as f:
      pickle.dump(simple_r_v5, f)
    with open(f"{model}/kappa/{model_name}_test_with_view_5.pickle", 'wb') as f:
      pickle.dump(simple_r_v6, f)


def two_shot_trainer(dataset, view, num_shots):
    args = arg_parse(dataset, view, num_shots)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)  
    start = time.time()
    
    for i in range(args.num_shots):
        model = "gcn"
        model_name = "Few_Shot_"+dataset+"_"+model + str(i)
        print("Shot : ",i)
        with open('./Two_shot_samples_views/'+dataset+'_view_'+str(view)+'_shot_'+str(i)+'_train','rb') as f:
            train_set = pickle.load(f)
        with open('./Two_shot_samples_views/'+dataset+'_view_'+str(view)+'_shot_'+str(i)+'_test','rb') as f:
            test_set = pickle.load(f)
        
        num_nodes = train_set[0]['adj'].shape[0]
        model_GCN = GCN(nfeat = num_nodes,
                        nhid = args.hidden,
                        nclass = args.num_classes,
                        dropout = args.dropout)
        train_dataset, val_dataset, threshold_value = cross_val.two_shot_loader(train_set, test_set, args)
        
        test_acc = train(args, train_dataset, val_dataset, model_GCN, threshold_value, model_name) # model_name+"_view_"+str(view)
        print("Test accuracy:"+str(test_acc))
        print('load data using ------>', time.time()-start)
