import torch

from tqdm import tqdm 
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from BinaryClassifier import Classifier

from utils import set_device

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def predict_fn(data_loader, model, device):
    model.eval()
    fin_outputs = []
    with torch.no_grad():
        for bi, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            X = batch
            X = X[0].to(device)

            outputs = model(X)
            _outputs = torch.nn.functional.softmax(outputs, dim=1)[:,1].data.cpu().numpy()

            fin_outputs.extend(_outputs.tolist())
            
    return fin_outputs

class XAI_DS(Dataset):
        def __init__(self, data_path, data_target, augmentation=None):
            super().__init__()
            self.data_path = data_path
            self.data_target = data_target
            self.augmentation = augmentation
        
        def __len__(self):
            return len(self.data_path)

        def __getitem__(self, idx):
            sample = torch.load(self.data_path[idx]).permute(2,0,1)
            if self.augmentation:
                input_tensor = self.augmentation(sample)
            else:
                 input_tensor = sample
            return input_tensor, torch.tensor(self.data_target[idx], dtype=torch.float)
        
def err_stat(pred_label, True_label, test_df, class_name):
    err_index = []
    count_err = {}
    class_name = class_name
    for i in range(len(pred_label)):
        if pred_label[i] != True_label[i]:
            err_index.append(i)
    name_err = []
    name_err_1 = []
    for i in err_index:
        name_err.append(test_df[i].split('\\')[-1].split('_')[0])
        name_err_1.append(test_df[i].split('\\')[-1])
    for j in class_name:
        count_err[j] = name_err.count(j)
    return count_err, name_err_1

def Binary_metrics(true_labels, pred_labels, mode='all'):
    temp = 0.
    if mode == 'accuracy':
        temp = accuracy_score(true_labels, pred_labels)
    if mode == 'precision':
        temp = precision_score(true_labels, pred_labels)
    if mode == 'recall':
        temp = recall_score(true_labels, pred_labels)
    if mode == 'f1':
        temp = f1_score(true_labels, pred_labels)
    if mode == 'roc_auc':
        temp = roc_auc_score(true_labels, pred_labels)
    if mode == 'all':
        temp = {}
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        # roc_auc = roc_auc_score(true_labels, pred_labels)
        temp['accuracy'] = accuracy
        temp['precision'] = precision
        temp['recall'] = recall
        temp['f1'] = f1
        # temp['roc_auc'] = roc_auc
    return temp

def make_label(list_fin_outputs):
    labels = []
    threshold = 0.50
    temp = 0
    for i in list_fin_outputs:
        if i > threshold:
            temp = 1 # Adv
        else:
            temp = 0 # Clean
        labels.append(temp)
    return labels

def plot_confusion_matrix(confusion_matrix, class_names):
        plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="YlGnBu", cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()
        plt.savefig('./pic-{}.png'.format('confusion_matrix'))
        
if __name__ == '__main__':
    weights = torch.load("./Model/CheckPoint/ckpt_best_30.pth")['net']
    Binary_classifier_XAI = Classifier()
    model = Binary_classifier_XAI
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to('cuda')

    model.load_state_dict(weights)
    
    df = pd.read_csv('./Dataset_No3.csv')
    train_df, val_temp = train_test_split(df, test_size=0.4, random_state=45)
    val_df, test_df = train_test_split(val_temp, test_size=0.5, random_state=45)

    val_transforms = transforms.Compose([transforms.Normalize([-3.8328e-05, -9.1511e-05, -6.0215e-05], [0.0114, 0.0207, 0.0108])])
    
    test_paths = [i for i in train_df["image"].values]
    test_target = train_df["label"].values
    ds_test = XAI_DS(test_paths, test_target, val_transforms)
    test_loader = torch.utils.data.DataLoader(ds_test , batch_size=32, shuffle=False)   
    preds = predict_fn(test_loader, model, set_device())

    pred_label = make_label(preds)
    True_label = list(test_target)

    dict_pred = Binary_metrics(True_label, pred_label , mode='all')
    err_, err_name = err_stat(pred_label, True_label, test_paths, class_name=['cw', 'norm', 'bim', 'pdg'])
    roc_auc = roc_auc_score(True_label, preds)

    confusion_matrix_ = confusion_matrix(True_label, pred_label)

    print(roc_auc, dict_pred, err_)
    class_names = ["Clean sample (0)", "Adversarial sample (1)"]
    plot_confusion_matrix(confusion_matrix_, class_names)
