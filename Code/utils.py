import torch
from torchvision import models, transforms
import cv2
import numpy as np
import torch.nn.functional as F

import random
import json

def set_device():
    device = ''
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

def set_model(model_name='resnet50', weights='ResNet50_Weights.IMAGENET1K_V2', device='cpu', eval_model=False):
    if model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        if eval_model:
            model = model.eval()
        model.to(device)

    elif model_name == 'resnet34':
        model = models.resnet34(weights=weights)
        if eval_model:
            model = model.eval()
        model.to(device)

    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        if eval_model:
            model = model.eval()
        model.to(device)

    elif model_name == 'resnet152':
        model = models.resnet152(weights=weights)
        if eval_model:
            model = model.eval()
        model.to(device)

    elif model_name == 'alexnet':
        model = models.alexnet(weights=weights)
        if eval_model:
            model = model.eval()
        model.to(device)

    elif model_name == 'inceptionv3':
        model = models.inception_v3(weights=weights)
        if eval_model:
            model = model.eval()
        model.to(device)

    elif model_name == 'wideresnet':
        model = models.wide_resnet50_2(weights=weights)
        if eval_model:
            model = model.eval()
        model.to(device)

    elif model_name == 'vgg11':
        model = models.vgg11(weights=weights)
        if eval_model:
            model = model.eval()
        model.to(device)

    elif model_name == 'vgg13':
        model = models.vgg13(weights=weights)
        if eval_model:
            model = model.eval()
        model.to(device)

    elif model_name == 'vit':
        model = models.vit_l_16(weights=weights)
        if eval_model:
            model = model.eval()
        model.to(device)
        
    return model, model_name

def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) 
    img = img[:, :, (2, 1, 0)]
    return img

def pre_processing(img, norm=False, device='cpu', grad_require=False):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    img = img / 255
    if norm:
        img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    # img = np.expand_dims(img, 0)
    img = np.array(img)
    img_tensor = torch.tensor(img, dtype=torch.float32, device=device)
    if grad_require:
        img_tensor.requires_grad = True
    return img_tensor

def show_prob(class_index_path_json, model, input, transform=False):
    transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
    )
    if transform:
        input = transform_normalize(input)
    labels_path = class_index_path_json
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)
    _pred = model(input)
    prob = F.softmax(_pred, dim=1)
    prediction_score, pred_label_idx = torch.topk(prob, 1)
    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    _target = torch.argmax(_pred)
    return predicted_label, prediction_score.squeeze().item(), _target

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def norm_zero_one(_array):
    min_val = np.min(_array)
    max_val = np.max(_array)
    return (_array - min_val) / (max_val - min_val)
