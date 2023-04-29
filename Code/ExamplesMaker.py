import torch

import torchattacks

# torch.backends.cudnn.deterministic = True

from tqdm import tqdm
from utils import load_img, pre_processing, set_device, set_model

from InterMaker import gbp_gen_, gradpp_cam_gen_

import numpy as np

from CW import cw_l2_attack

import os
import re
import random

def take_out_name(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    pattern = r'.*_.*'
    match = re.search(pattern, file_name)

    if match:
        extracted_string = match.group(0)
        temp = extracted_string
    else:
        print("No matching string found.")
        temp = 'None'
    return temp
            
def split_list_into_parts(lst, num_parts):
    avg_len = len(lst) // num_parts
    remainders = len(lst) % num_parts
    parts = []
    start = 0
    
    for i in range(num_parts):
        end = start + avg_len
        if remainders > 0:
            end += 1
            remainders -= 1
        parts.append(lst[start:end])
        start = end
    return parts

def get_pred(model, images):
    logits = model(images)
    _, pres = logits.max(dim=1)
    return pres.cpu()
  
def make_dict_label(path_img):
    dic = dict()
    list_name = os.listdir(path_img)
    count = 0
    for i in list_name:
      dic[i] = count
      count += 1
    return dic

def select_img(root_dir, num_img, num_sub_group):
    folder_list = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]

    selected_images = []
    selected_images_set = set()
    for folder in folder_list:
        image_list = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(('.png', '.jpg', '.JPEG'))]
        num_to_select = min(num_sub_group, len(image_list))
        sampled_images = random.sample(image_list, num_to_select)
        selected_images.extend(sampled_images)
        selected_images_set.update(sampled_images)

    num_additional_images = num_img- len(selected_images)

    while num_additional_images > 0:
        for folder in folder_list:
            image_list = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(('.png', '.jpg', '.JPEG'))]
            if len(image_list) > num_sub_group:
                remaining_images = list(set(image_list) - selected_images_set)
                num_to_select = min(num_additional_images, len(remaining_images))
                additional_images = random.sample(remaining_images, num_to_select)
                selected_images.extend(additional_images)
                selected_images_set.update(additional_images)
                num_additional_images -= num_to_select

            if num_additional_images <= 0:
                break
    return selected_images

if __name__ == '__main__':
    random.seed(48)
    model,_ = set_model('resnet50', device=set_device(), eval_model=True)
    #Set parameters as you need
    atk_1 = torchattacks.PGD(model, eps=1/255, alpha=2/225, steps=30, random_start=True)
    atk_1.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  
    atk_2 = torchattacks.BIM(model, eps=1/255, steps=30)
    atk_2.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  
    atk_3 = torchattacks.MIFGSM(model, eps=3/255, steps=30)
    atk_3.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  
    atk_4 = torchattacks.FGSM(model, eps=1/255)
    atk_4.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  
    atk_5 = torchattacks.AutoAttack(model, norm='Linf', eps=1/255, version='standard', n_classes=1000,verbose=True)
    atk_5.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  
    atk_6 = cw_l2_attack(model, images, labels, c=1, kappa=0, max_iter=500, learning_rate=0.01)
  
    attack_list = [atk_1, atk_2, atk_3, atk_4, atk_5, atk_6]
    name_attack_list = ['pgd', 'bim', 'mifgsm', 'fgsm', 'autoattack', 'cw']
    root_dir = './archive/imagenet-mini/train/'
    dic_labels = make_dict_label(root_dir) # like `` 'n01440764': 0 ''
    list_img = select_img(root_dir, 20000, 20) # Extract 20 images from 1000 folders (if a folder does not have enough 20 imgs, extract more from other folders), total 20000
    list_file_img = list()
    for i in list_img:
        list_file_img.append(i.split('\\')[0].split('/')[-1] + '/' + i.split('\\')[-1])

    count = 0
    for j in attack_list:
        for i in tqdm(list_file_img, desc="Images"):
            new_name = i.split('/')[0]
            i = root_dir + '/' + i
            name_pic = take_out_name(i)
            np_arr = load_img(i)

            tensor = pre_processing(np_arr, norm=True, device=set_device(), grad_require=False)
            tensor = tensor.unsqueeze(0)
            labels = get_pred(model, tensor) ## Labels of normal examples
            adv_img = j(tensor, labels)

            labels_adv = get_pred(model, adv_img)

            if (int(labels_adv) != int(labels)) and (int(labels) == dic_labels[new_name]):
                norm_ITG = gbp_gen_(model, tensor, _target=labels_adv)
                norm_GAM = gradpp_cam_gen_(model, tensor, labels_adv)

                ss = np.dstack((
                norm_ITG[:,:,0] * norm_GAM[0, :],
                norm_ITG[:,:,1] * norm_GAM[0, :],
                norm_ITG[:,:,2] * norm_GAM[0, :],))
                torch.save(torch.tensor(ss), './kkkk/{}_{}_{}_{}.pt'.format(name_attack_list[count],name_pic, int(labels_adv), int(labels)))
                count += 1
       
