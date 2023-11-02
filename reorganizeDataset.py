import pickle
import numpy as np
import scipy.io
import os

def SaveDataset():
    '''
    The given datasets have individual "LHASDSub1.mat" files for each subject. 
    The code in the repository works with two files: "ASD NC LH_edges" and 
    "ASD NC LH_labels." These files contain lists for N subjects. Each subject's 
    "edges" data has dimensions of 35x35x6, and the "labels" contain binary values, 
    either 1 or 0. This function is designed to convert these datasets, which are 
    stored in separate files for each subject, into a format that the repository can process.
    '''
    def GetOneGroup(dataset_path):        
        subjects = [scipy.io.loadmat(os.path.join(dataset_path, subject_name))['views'] for subject_name in os.listdir(dataset_path)][:150]
        #subjects = [os.path.join(dataset_path, subject_name)for subject_name in os.listdir(dataset_path)][:150]
        labels = [1 if 'ASD LH' in dataset_path else 0] * len(subjects)
        return subjects, labels

    name_split = dataset_name.split(' ')
    first_group_name = name_split[0] + ' ' + name_split[2]
    second_group_name = name_split[1] + ' ' + name_split[2]

    dataset_path = os.path.join(root_path, first_group_name)
    first_subjects, first_labels = GetOneGroup(dataset_path)
    
    dataset_path = os.path.join(root_path, second_group_name)
    second_subjects, second_labels = GetOneGroup(dataset_path)
 

    subjects = first_subjects + second_subjects
    labels = first_labels + second_labels
    
    with open(os.path.join(root_path, 'LH_ASDNC_edges'), 'wb') as handle:
        pickle.dump(subjects, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(root_path, 'LH_ASDNC_labels'), 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    
    '''
    dataset1_subjects, dataset1_labels = first_subjects[:50] + second_subjects[:50], first_labels[:50] + second_labels[:50]
    dataset2_subjects, dataset2_labels = first_subjects[50:100] + second_subjects[50:100], first_labels[50:100] + second_labels[50:100]
    dataset3_subjects, dataset3_labels = first_subjects[100:150] + second_subjects[100:150], first_labels[100:150] + second_labels[100:150]
    
    
    with open(os.path.join(root_path, relative_path + '_d1_edges'), 'wb') as handle:
        pickle.dump(dataset1_subjects, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(root_path, relative_path + '_d1_labels'), 'wb') as handle:
        pickle.dump(dataset1_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)       

    with open(os.path.join(root_path, relative_path + '_d2_edges'), 'wb') as handle:
        pickle.dump(dataset2_subjects, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(root_path, relative_path + '_d2_labels'), 'wb') as handle:
        pickle.dump(dataset2_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)   

    with open(os.path.join(root_path, relative_path + '_d3_edges'), 'wb') as handle:
        pickle.dump(dataset3_subjects, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(root_path, relative_path + '_d3_labels'), 'wb') as handle:
        pickle.dump(dataset3_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    '''



root_path = '/home/hcb/Desktop/MeasuringReproducibility/data/LH_ASDNC/'
dataset_name = 'ASD NC LH'
splits = dataset_name.split(' ')
relative_path = splits[2] + '_' + splits[0] + splits[1]
SaveDataset()

# kontrol
with open(root_path + 'LH_ASDNC_edges','rb') as f:
    multigraphs = pickle.load(f)    

with open(root_path + 'LH_ASDNC_labels','rb') as f:
    labels = pickle.load(f)    

print(labels)
print(len(multigraphs), multigraphs[0].shape)