
from torch_geometric.data import DataLoader
from models_sag import  Net
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import os
import pickle
import Analysis
import random
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def arg_parser(cv_current=0, model='none', num_shots=2, view=0, dataset='none', cv_number=5):
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--v', type=str, default=1)
    parser.add_argument('--data', type=str, default='Sample_dataset', choices = [ f.path[5:] for f in os.scandir("data") if f.is_dir() ])
    
    
    parser.add_argument("--brain_fold",
                      dest="brain_fold", type=int, default=1)
    parser.add_argument("--brain_view",
                      dest="brain_view", type=int, default=2)
    parser.add_argument("--num_features",
                      dest="num_features", type=int, default=35)
    parser.add_argument("--num_classes",
                      dest="num_classes", type=int, default=2)
    parser.add_argument('--seed', type=int, default=777,
                        help='seed')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, #0.001
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, #0.0001
                        help='weight decay')
    parser.add_argument('--nhid', type=int, default=256,
                        help='hidden size')
    parser.add_argument('--pooling_ratio', type=float, default=0.5,
                        help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.4, #0.4
                        help='dropout ratio')
    parser.add_argument('--dataset', type=str, default=dataset,
                        help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
    parser.add_argument('--epochs', type=int, default=50,  #7
                        help='maximum number of epochs')
    parser.add_argument('--num_shots', type=int, default=num_shots,  #100
                        help='nbr of shots')
    parser.add_argument('--patience', type=int, default=100,
                        help='patience for earlystopping')
    parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                        help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
    parser.add_argument('-best_f1', type=int, default=0, help='keeps the best f1 during training')
    parser.add_argument('-best_f1_changed', type=bool, default=False, help='')

    parser.add_argument('--cv_number', type=int, default=cv_number, help='total cv numbers')
    parser.add_argument('--cv_current', type=int, default=cv_current,
                        help='number of validation folds.')  
    parser.add_argument('--model', type=str, default='none', help='GNN architecture to train')
    parser.add_argument('--view', type=int, default=view,
                        help = 'view index in the dataset')

    args = parser.parse_args()

    return args

def test(model,loader, is_trained, model_name, args):
    args.device = device
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    #model.eval()
    with torch.no_grad():
        correct = 0.
        loss = 0.
        preds = []
        labels =[]
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss += F.nll_loss(out,data.y,reduction='sum').item()
            preds.append(int(pred))
            labels.append(int(data.y))

        result = {'prec': metrics.precision_score(labels, preds, average='binary'),
                'recall': metrics.recall_score(labels, preds, average='binary'),
                'acc': metrics.accuracy_score(labels, preds),
                'F1': metrics.f1_score(labels, preds, average="binary"),
                'specificity': metrics.recall_score(labels, preds, pos_label=0, average='binary'),
                'test loss': loss / len(loader)} # specificity is the recall of the negative class . You can reach it just setting the pos_label parameter
        print('TEST SCORES', result)

        if is_trained:
            simple_r = {'acc': (correct / len(loader.dataset))}

        if result['F1'] >= args.best_f1:
            args.best_f1_changed = True
            args.best_f1 = result['F1']     

            simple_r = {'labels':labels,'preds':preds}
            with open("./sag/Labels_and_preds/"+model_name+"_test.pickle", 'wb') as f: # Labels_and_preds
              pickle.dump(simple_r, f)

            w_dict = {"w": model.conv1.lin.weight}
            with open("SAG_W.pickle", 'wb') as f: # weights
              pickle.dump(w_dict, f)
              print("SAG Weights are saved.")
            print('SASASAVED')

            print('Best F1 Obtained: ', args.best_f1)

    return result,loss / len(loader.dataset)

def cv_benchmark(dataset, view, cv_number, cv_current, model, model_name):
        
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)  
    #name = "5Fold"
    cv = cv_number
    name = str(cv)+"Fold"
    args = arg_parser(cv_current=cv_current, model=model, view=view, dataset=dataset, cv_number=cv_number)

    
    args.device = 'cpu'
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    updated_name = "MainModel_"+name+"_"+dataset+ "_" + "sag"+"_view_"+str(view)
    #args = arg_parser()
    cv_name = updated_name+"_CV_"+str(args.cv_current)


    with open('Folds_processed'+str(cv)+"/"+dataset+'_view_'+str(view)+'_folds_'+ str(cv) +'_fold_'+str(args.cv_current)+'_train_pg','rb') as f:
        train_set = pickle.load(f)
    with open('Folds_processed'+str(cv)+"/"+dataset+'_view_'+str(view)+'_folds_'+ str(cv) +'_fold_'+str(args.cv_current)+'_test_pg','rb') as f:
        test_set = pickle.load(f)

    print("Size of training set:"+str(len(train_set)))
    print("Size of test set:"+str(len(test_set)))
    print("CV : ", args.cv_current)
    model = Net(args).to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_set,batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set,batch_size=args.batch_size, shuffle=True)

    training_loss = []
    for epoch in range(args.epochs):
        print('epoch : ',epoch)
        model.train()
        total_loss = 0
        tensor_preds = torch.empty(size=(len(train_loader), 2))  # added
        tensor_labels = torch.empty(size=(len(train_loader), 1)) # added
        preds = []
        labels = []
        for i, data in enumerate(train_loader):
            data = data.to(device)
            out = model(data)
            tensor_preds[i] = out      # added  
            tensor_labels[i] = data.y  # added
            loss = F.nll_loss(out, data.y)
            preds.append(int(out.max(dim=1)[1].detach().numpy()))
            labels.append(int(data.y.detach().numpy()))
            total_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        training_loss.append(total_loss.item())
        #print("Training loss:{}".format(total_loss.item()))
        simple_r = {'labels':labels,'preds':preds}
        result = {'prec': metrics.precision_score(labels, preds, average='binary'),
                'recall': metrics.recall_score(labels, preds, average='binary'),
                'acc': metrics.accuracy_score(labels, preds),
                'F1': metrics.f1_score(labels, preds, average="binary"),
                'specificity': metrics.recall_score(labels, preds, pos_label=0, average='binary'),
                'train loss': loss / len(train_loader)} # specificity is the recall of the negative class . You can reach it just setting the pos_label parameter
        print('TRAIN SCORES', result)

        test_acc,test_loss = test(model,test_loader, 1, cv_name, args)

        if epoch == args.epochs-1:
            Analysis.is_trained = True
        #train_preds = tensor_preds.cpu().detach().numpy()   # added
        #train_labels = tensor_labels.cpu().detach().numpy() # added
        val_acc,val_loss = test(model,val_loader, 0, cv_name, args)
        #print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
        if args.best_f1_changed:
            los_p = {'loss':training_loss}

            with open("./sag/training_loss/Training_loss_"+cv_name+".pickle", 'wb') as f: # loss
                pickle.dump(los_p, f)
            with open("./sag/Labels_and_preds/"+model_name+"_train.pickle", 'wb') as f:
                pickle.dump(simple_r, f)
    
            path = './sag/weights/W_'+cv_name+'.pickle'
            os.rename('SAG_W.pickle', path)
            torch.save(model.state_dict(), "./sag/models/SAG_"+cv_name+".pt") # models
            args.best_f1_changed = False

    args.best_f1 = 0

    #print("Test accuracy:{}".format(test_acc))

def two_shot_trainer(dataset, view, num_shots):
    
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)  
    
    args = arg_parser(num_shots)
    args.device = 'cpu'
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    updated_name = "Few_Shot_"+ dataset + "_sag"+"_view_"+str(view)
    for cv_n in range(args.num_shots):
        cv_name = updated_name+"_"+str(cv_n)
        with open('Two_shot_processed/'+dataset+'_view_'+str(view)+'_shot_'+str(cv_n)+'_train_pg','rb') as f:
            train_set = pickle.load(f)
        with open('Two_shot_processed/'+dataset+'_view_'+str(view)+'_shot_'+str(cv_n)+'_test_pg','rb') as f:
            test_set = pickle.load(f)
        print("Size of training set:"+str(len(train_set)))
        print("Size of test set:"+str(len(test_set)))
        print("Run : ",cv_n)
        model = Net(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(test_set,batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set,batch_size=args.batch_size, shuffle=True)
    
        training_loss = []
        for epoch in range(args.epochs):
            print('epoch : ',epoch)
            model.train()
            
            total_loss = 0
            tensor_preds = torch.empty(size=(len(train_loader), 2))  # added
            tensor_labels = torch.empty(size=(len(train_loader), 1)) # added
            for i, data in enumerate(train_loader):
                data = data.to(device)
                out = model(data)
                tensor_preds[i] = out      # added  
                tensor_labels[i] = data.y  # added
                pred = out.max(dim=1)[1]
                loss = F.nll_loss(out, data.y)
                total_loss += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            if epoch == args.epochs-1:
                Analysis.is_trained = True
            training_loss.append(total_loss.item())
            #print("Training loss:{}".format(total_loss.item()))
            #train_preds = tensor_preds.cpu().detach().numpy()   # added
            #train_labels = tensor_labels.cpu().detach().numpy() # added
            val_acc,val_loss = test(model,val_loader, 0, cv_name, args)
            #print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
    
        los_p = {'loss':training_loss}
        with open("./sag/training_loss/Training_loss_"+cv_name+".pickle", 'wb') as f:
          pickle.dump(los_p, f)
    
    
        path = './sag/weights/W_'+cv_name+'.pickle'
    
        if os.path.exists(path):
            os.remove(path)
            
        os.rename('SAG_W.pickle', path)
        torch.save(model.state_dict(),"./sag/models/SAG_"+cv_name+".pt")
        test_acc,test_loss = test(model,test_loader, 1, cv_name, args)
        print("Test accuarcy:{}".format(test_acc))
  