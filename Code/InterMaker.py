import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
import json

from matplotlib.colors import LinearSegmentedColormap


from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel

from captum.attr import Saliency, GuidedGradCam, GuidedBackprop, LRP, IntegratedGradients
from captum.attr import visualization as viz

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import EigenCAM, EigenGradCAM, LayerCAM
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, GradCAMElementWise
from pytorch_grad_cam import AblationCAM, RandomCAM, FullGrad ,HiResCAM, XGradCAM
from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel

from ScoreCAM.cam.scorecam import ScoreCAM

from utils import set_device, pre_processing

def numpy_to_image(arr):
    arr = arr * 255
    arr = np.uint8(arr)
    img = Image.fromarray(np.transpose(arr, (1,2,0)))
    return img

def min_max_scaler(tensor):
    tensor = np.array(tensor)
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    norm_tensor = (tensor - min_val) / (max_val - min_val)
    return norm_tensor

def visualize_gen(attr_array, orig_array):
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                             [(0, '#000000'),
                                              (0.25, '#068f41'),
                                              (1, '#068f41')], N=224)

    _ = viz.visualize_image_attr(attr_array,
                            orig_array,
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=False,
                             sign='positive',
                             outlier_perc=0.1,
                             fig_size = (3.12, 3.12))
    return _[0]

def get_pred(model, images):
    logits = model(images)
    _, pres = logits.max(dim=1)
    return pres.cpu()

def make_target(img_path, model, labels_path, Adv_Norm=0):
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)
    tensor = img_path.unsqueeze(0).to(set_device())
    __target = get_pred(model, tensor)
    # print(__target)
    # _target = model(tensor)
    # output = F.softmax(_target, dim=1)
    # prediction_score, pred_label_idx = torch.topk(output, 1)
    # pred_label_idx.squeeze_()
    # predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    # print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    return int(__target)

def ITG(img_path, model, _target):
    _input = torch.load(img_path)
    Saliency_maps = IntegratedGradients(model, multiply_by_inputs=True)
    _input = torch.tensor(_input).to(set_device())
    _input.requires_grad = True
    S_MS = Saliency_maps.attribute(_input.unsqueeze(0), target=_target)
    S_MS = np.transpose(S_MS.squeeze().cpu().detach().numpy(), (1,2,0))
    return S_MS

def GGC(img_path, model):
    _input = torch.load(img_path)
    Saliency_maps = GuidedGradCam(model, model.layer4)
    _target = make_target(_input, model, labels_path='./imagenet_class_index.json')
    _input.requires_grad = True
    S_MS = Saliency_maps.attribute(_input, target=_target)
    S_MS = np.transpose(S_MS.squeeze().cpu().detach().numpy(), (1,2,0))
    return S_MS

def SaliencyMaps(img_path, model):
    _input = torch.load(img_path)
    Saliency_maps = Saliency(model)
    _target = make_target(_input, model, labels_path='./imagenet_class_index.json')
    _input = torch.tensor(_input).to(set_device())
    _input.requires_grad = True
    S_MS = Saliency_maps.attribute(_input.unsqueeze(0), target=_target)
    S_MS = np.transpose(S_MS.squeeze().cpu().detach().numpy(), (1,2,0))
    return S_MS

def GBP(img_path, model):
    _input = torch.load(img_path)
    Saliency_maps = GuidedBackprop(model)
    _target = make_target(_input, model, labels_path='./imagenet_class_index.json')
    _input = torch.tensor(_input).to(set_device())
    _input.requires_grad = True
    S_MS = Saliency_maps.attribute(_input.unsqueeze(0), target=_target)
    S_MS = np.transpose(S_MS.squeeze().cpu().detach().numpy(), (1,2,0))
    return S_MS

def gradpp_cam_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0)
    target = ClassifierOutputTarget(_target)
    cam = GradCAMPlusPlus(model, target_layers = [model.layer4[-1]], use_cuda= True)
    grayscale_cam = cam(tensor, targets=[target])
    return grayscale_cam

def eigen_cam_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0)
    target = ClassifierOutputTarget(_target) 
    cam = EigenCAM(model, target_layers = [model.layer4[-1]], use_cuda= True)
    grayscale_cam = cam(tensor, targets=[target])
    return grayscale_cam

def layer_cam_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0) 
    target = ClassifierOutputTarget(_target)
    cam = LayerCAM(model, target_layers = [model.layer4[-1]], use_cuda= True)
    grayscale_cam = cam(tensor, targets=[target])
    return grayscale_cam

def eigengrad_cam_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0)
    target = ClassifierOutputTarget(_target)
    cam = EigenGradCAM(model, target_layers = [model.layer4[-1]], use_cuda= True)
    grayscale_cam = cam(tensor, targets=[target])
    return grayscale_cam

def XGradCAM_cam_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0)
    target = ClassifierOutputTarget(_target)
    cam = XGradCAM(model, target_layers = [model.layer4[-1]], use_cuda= True)
    grayscale_cam = cam(tensor, targets=[target])
    return grayscale_cam

def gbp_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0)
    cam = GuidedBackpropReLUModel(model, use_cuda= True)
    grayscale_cam = cam(tensor, _target)
    return grayscale_cam

def fullgrad_cam_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0)
    target = ClassifierOutputTarget(_target) 
    cam = FullGrad(model, target_layers = [model.layer4[-1]], use_cuda= True)
    grayscale_cam = cam(tensor, targets=[target])
    return grayscale_cam

def hires_cam_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0)
    target = ClassifierOutputTarget(_target) 
    cam = HiResCAM(model, target_layers = [model.layer4[-1]], use_cuda= True)
    grayscale_cam = cam(tensor, targets=[target])
    return grayscale_cam

def random_cam_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0)
    target = ClassifierOutputTarget(_target) 
    cam = RandomCAM(model, target_layers = [model.layer4[-1]], use_cuda= True)
    grayscale_cam = cam(tensor, targets=[target])
    return grayscale_cam

def gradcamelementwise_cam_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0)
    target = ClassifierOutputTarget(_target) 
    cam = GradCAMElementWise(model, target_layers = [model.layer4[-1]], use_cuda= True)
    grayscale_cam = cam(tensor, targets=[target])
    return grayscale_cam

def grad_cam_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0)
    target = ClassifierOutputTarget(_target)
    cam = GradCAM(model, target_layers = [model.layer4[-1]], use_cuda= True)
    grayscale_cam = cam(tensor, targets=[target])
    return grayscale_cam

def ablation_cam_gen_(model, img_path, _target):
    # tensor = torch.load(img_path)
    tensor = img_path
    tensor.to(set_device())
    if int(len(tensor.shape)) > 3:
        tensor = tensor
    else:
        tensor = tensor.unsqueeze(0)
    target = ClassifierOutputTarget(_target)
    cam = AblationCAM(model, target_layers = [model.layer4[-1]], use_cuda= True)
    grayscale_cam = cam(tensor, targets=[target])
    return grayscale_cam

def Score_cam_gen(model, img_tensor):
    img_tensor = torch.load(img_tensor).to(set_device())
    resnet_model_dict = dict(type='resnet50', arch=model, layer_name='layer4',input_size=(224, 224))
    resnet_scorecam = ScoreCAM(resnet_model_dict)
    if len(img_tensor.shape) > 3:
        img_tensor = img_tensor
    else:
        img_tensor = img_tensor.unsqueeze(0)
    scorecam_map = resnet_scorecam(img_tensor)
    return np.array(scorecam_map.squeeze(0)[0, :, : ].cpu())

def gradcam_common(grayscale_cam, grayscale_common):
    if len(grayscale_common.shape) >= 3:
        grayscale_common = np.mean(grayscale_common, axis=2)
    return np.multiply(grayscale_common, grayscale_cam)

def GradientInput(img, model):
    img = np.transpose(img, (1,2,0))
    img = img * 255
    inputs = pre_processing(img, norm=True, device=set_device(), grad_require=True)
    grads = []

    output = model(inputs)
    output = F.softmax(output, dim=1)
    
    target_label_idx = torch.argmax(output, 1).item()
    index = np.ones((output.size()[0], 1)) * target_label_idx
    index = torch.tensor(index, device=set_device(), dtype=torch.int64)
    output = output.gather(1, index)
    
    model.zero_grad()
    output.backward()
    
    gradient = inputs.grad.detach().cpu().numpy()[0]
    grads.append(gradient)
    grads = np.array(grads)
    grads = np.transpose(grads[0], (1, 2, 0))
    
    GradientInput = np.multiply(img, grads)
        
    return GradientInput
    
def integrated_gradients(img_path, model, batch_blank_type='zero', iterations=50, flag = 0):
    img_path = np.transpose(img_path, (2,1,0))
    batch_x = img_path * 255
    IG = []
    if flag == 0:
        norm = True
    else:
        norm = False

    for _ in range(2):
        grads = []
        x = []
        
        if batch_blank_type == 'zero':
            batch_blank = np.zeros(shape=batch_x.shape) 
        if batch_blank_type == 'one':
            batch_blank = np.ones(shape=batch_x.shape)
        if batch_blank_type == 'rand':
            batch_blank = 255.0 * np.random.random(batch_x.shape)
            
        for i in range(0, iterations + 1):
            k = i / iterations
            x.append(batch_blank + k * (batch_x - batch_blank))
            
        for input in x:
            input = pre_processing(input, norm=norm, device=set_device(), grad_require=True)
            output = model(input)

            output = F.softmax(output, dim=1)
            target_label_idx = torch.argmax(output, 1).item()
            index = np.ones((output.size()[0], 1)) * target_label_idx
            index = torch.tensor(index, device=set_device(), dtype=torch.int64)
            output = output.gather(1, index)
            
            model.zero_grad()
            output.backward()
            
            gradient = input.grad.detach().cpu().numpy()[0]
            grads.append(gradient)

        grads = np.array(grads)
        avg_grads = np.average(grads[:-1], axis=0)
        avg_grads = np.transpose(avg_grads, (1, 2, 0))
        delta_X = (pre_processing(batch_x, norm=norm, device=set_device(),  grad_require=True) - pre_processing(batch_blank, norm=norm, device=set_device(), grad_require=True)).detach().squeeze(0).cpu().numpy()
        integrated_grad = np.multiply(np.transpose(delta_X, (1, 2, 0)), avg_grads)
        IG.append(integrated_grad)
        if batch_blank_type == 'zero':
            break
        elif batch_blank_type == 'one':
            break
        grads = []
    F_IG = np.average(np.array(IG), axis=0)
    return F_IG
