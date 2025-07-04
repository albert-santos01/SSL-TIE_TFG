import cv2
import librosa
import numpy as np
import shutil
import json
import sys

import scipy
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchaudio
import torchaudio.transforms as audio_T
from torch.autograd import Variable
from torch import nn

from models.model import AVENet
import subprocess

import os
import wandb
from PIL import Image
from datetime import datetime, timedelta
from opts import get_arguments

from abc import ABC, abstractmethod

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def normalize_img(value, vmax=None, vmin=None):
    '''
    Normalize heatmap
    '''
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value

def vis_loader(image,spec,index):
    img = image[0].cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    print("image",img.shape,img)
    print(spec.shape)
    spec = spec[0].cpu().numpy()
    spec = spec.squeeze(0)
    print("spec", spec.shape,spec)

    img = normalize_img(img)
    spec = normalize_img(spec)

    img_vis = (img * 255).astype(np.uint8)
    spec_vis = (spec * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_vis)
    spec_pil = Image.fromarray(spec_vis)

    img_pil.save(f'images_for_study/{index}_image.png')
    spec_pil.save(f'images_for_study/{index}_spec.png')
    
    raise ValueError("Stop")


def vis_matchmap(matchmap, img_array, video_name, epoch, save_dir, args, img_size=224):
    '''
    Visualization for the matchmap video
    matchmap  shape [14,14,128]
    img_array shape [3, H, W]
    '''

    img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img,(img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    
    video_path = os.path.join(save_dir, f"{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 1.0, (img_size, img_size))

    for i in range(np.shape(matchmap)[2]):
        matchmap_i = matchmap[:, :, i]
        matchmap_i = cv2.resize(matchmap_i, dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
        matchmap_i = normalize_img(-matchmap_i)

        for x in range(matchmap_i.shape[0]):
            for y in range(matchmap_i.shape[1]):
                matchmap_i[x][y] = (matchmap_i[x][y] * 255).astype(np.uint8)

        matchmap_i = matchmap_i.astype(np.uint8)
        matchmap_i_img = cv2.applyColorMap(matchmap_i, cv2.COLORMAP_JET)
        matchmap_i_img = cv2.addWeighted(matchmap_i_img, 0.5, img, 0.5, 0)
        matchmap_i_img_BGR = cv2.cvtColor(matchmap_i_img, cv2.COLOR_RGB2BGR)
        out.write(matchmap_i_img_BGR)

        out.release()

    if args.use_wandb:
        wandb.log({"video": wandb.Video(video_path), "epoch": epoch})

def vis_heatmap_bbox(heatmap_arr, img_array, img_name=None, bbox=None, ciou=None,  testset=None, img_size=224, save_dir=None ):
    '''
    visualization for both image with heatmap and boundingbox if it is available
    heatmap_array shape [1,1,14,14]
    img_array     shape [3 , H, W]
    '''
    if bbox == None:
        img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img,(img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heatmap = cv2.resize(heatmap_arr[0,0], dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
        heatmap = normalize_img(-heatmap)

        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                heatmap[x][y] = (heatmap[x][y] * 255).astype(np.uint8)
        heatmap = heatmap.astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_on_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
        
        # return np.array(heatmap_on_img)
        heatmap_on_img_BGR = cv2.cvtColor(heatmap_on_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_dir , heatmap_on_img_BGR )



    # Add comments
    else:
        img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        ori_img = img
        img = cv2.resize(img,(img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heatmap = cv2.resize(heatmap_arr[0,0], dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
        heatmap = normalize_img(-heatmap)

        # bbox = False
        if bbox:
            for box in bbox:
                lefttop = (box[0], box[1])
                rightbottom = (box[2], box[3])
                img = cv2.rectangle(img, lefttop, rightbottom, (0, 0, 255), 1)

        # img_box = img
        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                heatmap[x][y] = (heatmap[x][y] * 255).astype(np.uint8)
        heatmap = heatmap.astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_on_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

        # if ciou:
        #     cv2.putText(heatmap_on_img, 'IoU:' + '%.4f' % ciou , org=(25, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        #                fontScale=0.5, color=(255,255,255), thickness=1)

        if save_dir:
            save_dir = save_dir + '/heat_img_vis/' 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            heatmap_on_img_BGR = cv2.cvtColor(heatmap_on_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_dir +'/' + img_name + '_' + '%.4f' % ciou + '.jpg', heatmap_on_img_BGR )
        

        # save_ori_img = True
        # if save_ori_img:
        #     save_dir = save_dir + '/../' + '/ori_img/'
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     cv2.imwrite(save_dir +'/' + img_name  + '.jpg', ori_img )

        
        return np.array(heatmap_on_img)
        

def vis_masks(masks, img_array, img_name=None, bbox=None, ciou=None,  testset=None, img_size=224, save_dir=None ):
    '''
    visualization for both image with heatmap and boundingbox if it is available
    heatmap_array shape [1,1,14,14]
    img_array     shape [3 , H, W]
    '''
    # if bbox == None:
    #     pass
    #     img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    #     img = cv2.resize(img,(img_size, img_size))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     heatmap = cv2.resize(heatmap_arr[0,0], dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    #     heatmap = normalize_img(-heatmap)

    #     for x in range(heatmap.shape[0]):
    #         for y in range(heatmap.shape[1]):
    #             heatmap[x][y] = (heatmap[x][y] * 255).astype(np.uint8)
    #     heatmap = heatmap.astype(np.uint8)
    #     heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #     heatmap_on_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
        
    #     return np.array(heatmap_on_img)


    # Add comments
    # else:
    img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img,(img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # heatmap = cv2.resize(heatmap_arr[0,0], dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    # heatmap = normalize_img(-heatmap)

    if bbox:
        for box in bbox:
            lefttop = (box[0], box[1])
            rightbottom = (box[2], box[3])
            img = cv2.rectangle(img, lefttop, rightbottom, (128, 0, 128), 1)


    inter_img = img.copy()
    gt_img    = img.copy()
    pred_img  = img.copy()

    inter_mask, gt_mask, pred_mask, _ = masks

    inter_img[inter_mask] = (255,255,0)
    gt_img[gt_mask] = (0,255,0)
    pred_img[pred_mask] = (0,0,255)

    img = cv2.addWeighted(inter_img, 0.85, img, 0.15, 0 )      
    img = cv2.addWeighted(gt_img, 0.85, img, 0.15, 0 )
    img = cv2.addWeighted(pred_img, 0.85, img, 0.15, 0 )
    


    # img_box = img
    # for x in range(heatmap.shape[0]):
    #     for y in range(heatmap.shape[1]):
    #         heatmap[x][y] = (heatmap[x][y] * 255).astype(np.uint8)
    # heatmap = heatmap.astype(np.uint8)
    # heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmap_on_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

    if ciou:
        cv2.putText(img, 'IoU:' + '%.4f' % ciou , org=(25, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, color=(255,255,255), thickness=1)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_dir +'/' + img_name + '.jpg', img_BGR )
    
        # return np.array(heatmap_on_img)



def max_norm(p, version='torch', e=1e-5):
    if version is 'torch':
        if p.dim() == 3:
            C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
            min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
        elif p.dim() == 4:
            N, C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
            min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
    elif version is 'numpy' or version is 'np':
        if p.ndim == 3:
            C, H, W = p.shape
            p[p<0] = 0
            max_v = np.max(p,(1,2),keepdims=True)
            min_v = np.min(p,(1,2),keepdims=True)
            p[p<min_v+e] = 0
            p = (p-min_v-e)/(max_v+e)
        elif p.ndim == 4:
            N, C, H, W = p.shape
            p[p<0] = 0
            max_v = np.max(p,(2,3),keepdims=True)
            min_v = np.min(p,(2,3),keepdims=True)
            p[p<min_v+e] = 0
            p = (p-min_v-e)/(max_v+e)
    return p

def tensor2img(img, imtype=np.uint8, resolution=(224,224), unnormalize=True):

    img = img.cpu()
    if len(img.shape) == 4:
        img = img[0]
        
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    
    if unnormalize:
        img = img * std[:, None, None] + mean[:, None, None]
    
    img_numpy = img.numpy()
    img_numpy *= 255.0
    img_numpy = np.transpose(img_numpy, (1,2,0))
    img_numpy = img_numpy.astype(imtype)
    
    if resolution:
        img_numpy = cv2.resize(img_numpy, resolution) 

    return img_numpy

def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
    TODO: We have to reorder dimensions
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]), simtype)
    return S

def calc_recalls(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
	Computes recall at 1, 5, and 10 given encoded image and audio outputs.
	"""
    S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype)
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def computeMatchmap(I, A):
    """
    Computes the 3rd order tensor of matchmap between image and audio.
        I (C x H x W) 
        A  (C x T)
        Assumption: the channel dimension is already normalized
    """
    assert(I.dim() == 3)
    assert(A.dim() == 2)

    matchmap = torch.einsum('ct, chw -> thw', A, I)
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError('Unknown similarity type: %s' % simtype)

def sampled_margin_rank_loss(image_outputs, audio_outputs, margin=1., simtype='MISA'):
    """
    From DAVENet - Harwath et al. 2018
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    B = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(B):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i: #Try to sample a different image
            I_imp_ind = np.random.randint(0, B)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, B)
        anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i]), simtype)
        Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i]), simtype)
        Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / B
    return loss

def tensor_memory_MB(tensor, name):
    """
    Returns the memory size of a tensor in megabytes (MB).
    
    Args:
        tensor (torch.Tensor): The tensor to inspect.
    
    Returns:
        float: Size in MB.
    """
    size_bytes = tensor.nelement() * tensor.element_size()
    size_mb = size_bytes / (1024 ** 2)
    print(f"Size of {name}: {size_mb:.2f} MB")
    return size_mb

def volumemap_sim(volume_matrix, simtype='MISA',nFrames=None):
    """
    Computes the similarity scoring for all the combinations B x B
        volume_matrix (B x B x T x H x W)
        simtype MISA | SISA | SIMA
        nFrames None | (B x B) containing the n of unmasked
        Returns (B x B) similarity matrix
    """
    assert(volume_matrix.dim() == 5)
                
    if simtype== 'MISA':
        #MI
       volume_matrix, _ = volume_matrix.max(-1) # x spatial
       volume_matrix, _ = volume_matrix.max(-1) # y spatial
        #MISA
       if nFrames is None:
            volume_matrix = volume_matrix.mean(-1) # t temporal
       else: # Truncated Volume
            volume_matrix = volume_matrix.sum(dim=(2)) / nFrames  # To avoid bias in truncation
       
    elif simtype == 'SISA':
        volume_matrix = volume_matrix.mean(-1)
        volume_matrix = volume_matrix.mean(-1)
        if nFrames is None:
            volume_matrix = volume_matrix.mean(-1)
        else: # Truncated Volume
            volume_matrix = volume_matrix.sum(dim=(2)) / nFrames

    elif simtype == 'SIMA':
        volume_matrix,_ = volume_matrix.max(2) # MA #truncated frames are not included
        volume_matrix = volume_matrix.mean(-1)
        volume_matrix = volume_matrix.mean(-1) #SIMA
    else:
        raise ValueError('Unknown similarity type: %s' % simtype)

    assert(volume_matrix.dim() == 2)
    return volume_matrix


def similarity_matrix_bxb(img_outs, aud_outs,temp=0.07,simtype='MISA'):
    """
        img_outs (B x C x H x W) 
        aud_outs  (B x C x T)
        Assumption: the channel dimension is already normalized
        Returns a B x B similarity matrix: (exp(s_ij / temp))
    """
    assert(img_outs.dim() == 4)
    assert(aud_outs.dim() == 3)
    s_outs = torch.einsum('bct, pchw -> bpthw', aud_outs, img_outs)

    s_outs = volumemap_sim(s_outs,simtype)
    
    s_outs = torch.exp(s_outs / temp)

    return s_outs

def similarity_matrix_bxb_truncated(img_outs, aud_outs, nFrames, temp=0.07, simtype='MISA'):
    """
        img_outs (B x C x H x W) 
        aud_outs  (B x C x T)
        nFrames: (B x 1) just for truncating the audio
        Assumption: the channel dimension is already normalized
        For the similarity, 
        Returns a B x B similarity matrix: (exp(s_ij / temp))
    """
    assert(img_outs.dim() == 4)
    assert(aud_outs.dim() == 3)
    B = img_outs.size(0)
    T = aud_outs.size(2)
    device= aud_outs.device

    s_outs = torch.einsum('bct, pchw -> bpthw', aud_outs, img_outs)
    
    nFrames = torch.tensor(nFrames,device=device)

    mask = torch.arange(T,device=device).view(1,1,T,1,1) < nFrames.view(B,1,1,1,1)

    s_outs_masked = volumemap_sim(s_outs * mask, simtype, nFrames=nFrames.unsqueeze(-1))

    s_outs_masked = torch.exp(s_outs_masked / temp)

    return s_outs_masked


def infoNCE_loss_LVS(image_outputs, audio_outputs, args):
    """
        images_outputs (B x C x H x W) 
        audio_outputs  (B x C x T)
        Assumption: the channel dimension is already normalized
        Returns the InfoNCE loss considering the background as a negative
    """
    B = image_outputs.shape[0]
    epsilon  =  args.epsilon
    epsilon2 = args.epsilon2
    tau = 0.03
    device = image_outputs.device
    mask = ( 1 -100 * torch.eye(B,B)).to(device)

    m = nn.Sigmoid()
    # avgpool = nn.AdaptiveMaxPool2d((1, 1)) 

    Sii = torch.einsum('bchw, bct -> bthw',image_outputs, audio_outputs).unsqueeze(1) # [B,1,T,H,W]
    Sij = torch.einsum('bct, pchw -> bpthw', audio_outputs, image_outputs) # [B,B,T,H,W]
    # Sij_avg = avgpool(Sij).view(B,B)  # [B,B]


    Pos = m((Sii - epsilon)/tau)  # Positives [B,1,T,H,W]
    if args.tri_map:
        Pos2 = m((Sii - epsilon2)/tau)
        Neg = 1 - Pos2
    else:
        Neg = 1 - Pos

    Pos_all = m((Sij - epsilon)/tau) # Foregrounds Sij [B,B,T,H,W]
    easyNegs = ((Pos_all * Sij).reshape(*Sij.shape[:2],-1).sum(-1) / Pos_all.reshape(*Pos_all.shape[:2],-1).sum(-1) ) * mask
    sim = easyNegs #[B,B]

    sim1 = (Pos * Sii).reshape(*Sii.shape[:2],-1).sum(-1) / (Pos.reshape(*Pos.shape[:2],-1).sum(-1)) # Pi Foregrounds Sii [B,1]
    sim2 = (Neg * Sii).reshape(*Sii.shape[:2],-1).sum(-1) / (Neg.reshape(*Neg.shape[:2],-1).sum(-1)) # Hard negatives Backgrounds [B,1]

    if args.Neg:
        logits = torch.cat((sim1,sim,sim2),1)/args.temperature  #[B,B+2]
    else:
        logits = torch.cat((sim1,sim),1)/args.temperature
    
    criterion = nn.CrossEntropyLoss()
    target = torch.zeros(logits.shape[0]).to(device, non_blocking=True).long()
    loss =  criterion(logits, target)

    return loss

    




def infoNCE_loss(image_outputs, audio_outputs,args,return_S=False,nFrames=None):
    """
        images_outputs (B x C x H x W) 
        audio_outputs  (B x C x T)
        nFrames: (B x 1) just for truncating the audio 
        Assumption: the channel dimension is already normalized
        Returns the InfoNCE loss
    """
    if args.LVS:
        return infoNCE_loss_LVS(image_outputs,audio_outputs,args)
    if args.big_temp_dim:
        assert (audio_outputs.size(2) > 70)

    assert (len(image_outputs.shape) == 4)
    assert (len(audio_outputs.shape) == 3)


    B = image_outputs.size(0)
    mask = torch.eye(B, device=image_outputs.device)
    if nFrames is not None:
        sims = similarity_matrix_bxb_truncated(image_outputs, audio_outputs, nFrames= nFrames, temp=args.temperature, simtype=args.simtype)
    else:
        sims =  similarity_matrix_bxb(image_outputs, audio_outputs,args.temperature,args.simtype)
    pos = sims * mask
    neg = sims * (1 - mask)

    # This iterates the images against their negative audios...
    loss_v_a = -torch.log(pos.sum(dim=1) / (pos.sum(dim=1) + neg.sum(dim=1))).mean() / 2 
    # This iterates the audios against their negative images...
    loss_a_v = -torch.log(pos.sum(dim=0) / (pos.sum(dim=0) + neg.sum(dim=0))).mean() / 2
    loss = loss_v_a + loss_a_v

    if return_S:
        return loss, sims
    else:
        return loss

import torch

def topk_accuracy(sims, k=5):
    """
    Computes the Top-k accuracy for image-to-audio and audio-to-image retrieval.

    Args:
        sims: Similarity matrix (B x B) 
        k: The value of k for top-k accuracy.
        args: Additional arguments (e.g., similarity metric settings).

    Returns:
        acc_v_a: Top-k accuracy for image-to-audio retrieval.
        acc_a_v: Top-k accuracy for audio-to-image retrieval.
    """
    B = sims.size(0)


    # Ground truth indices (diagonal matches)
    ground_truth = torch.arange(B, device=sims.device)

    # ---- Image-to-Audio Retrieval ----
    # Sort each row (images) by similarity scores in descending order
    _, retrieved_audios = sims.topk(k, dim=1)
    
    # Check if the correct audio (diagonal) is within the top-k retrieved
    correct_v_a = (retrieved_audios == ground_truth.view(-1, 1)).any(dim=1).float()

    # ---- Audio-to-Image Retrieval ----
    # Sort each column (audios) by similarity scores in descending order
    _, retrieved_images = sims.t().topk(k, dim=1)

    # Check if the correct image (diagonal) is within the top-k retrieved
    correct_a_v = (retrieved_images == ground_truth.view(-1, 1)).any(dim=1).float()

    # Compute mean accuracy
    acc_v_a = correct_v_a.mean().item()
    acc_a_v = correct_a_v.mean().item()

    return acc_v_a, acc_a_v

def topk_accuracies(sims, ks=[1, 5, 10]):
    """
    Computes the Top-k accuracy (Recall@k) for multiple k values for image-to-audio and audio-to-image retrieval.

    Args:
        sims: Similarity matrix (B x B)
        ks: List of k values for which Recall@k should be computed.

    Returns:
        results: Dictionary containing both Audio and Image Recall@k values.
    """
    B = sims.size(0)
    max_k = max(ks)  # Get the maximum k needed

    # Ground truth indices (diagonal matches)
    ground_truth = torch.arange(B, device=sims.device)

    # ---- Compute Top-k indices once ----
    _, retrieved_images = sims.topk(max_k, dim=1)  # Audio-to-Image (Retrieve images for audio queries)
    _, retrieved_audios = sims.t().topk(max_k, dim=1)  # Image-to-Audio (Retrieve audios for image queries)

    # ---- Compute Recall@k in a single loop ----
    results = {}
    for k in ks:
        acc_a_v = (retrieved_images[:, :k] == ground_truth.view(-1, 1)).any(dim=1).float().mean().item()  # Audio-to-Image
        acc_v_a = (retrieved_audios[:, :k] == ground_truth.view(-1, 1)).any(dim=1).float().mean().item()  # Image-to-Audio

        results[f"A_r{k}"] = acc_a_v
        results[f"I_r{k}"] = acc_v_a

    return results

class GetSampleFromJson:
    def __init__(self, json_file, local_dir):
        self.json_file = json_file
        self.local_dir = local_dir
        with open(json_file, 'r') as f:
            data_json = json.load(f)
        self.data = data_json['data']
        self.image_base_path = data_json['image_base_path']
        self.audio_base_path = data_json['audio_base_path']
        self._init_transforms()
        self.AmplitudeToDB = audio_T.AmplitudeToDB()
        self.audio_conf = {}
        self.windows = {'hamming': scipy.signal.hamming,
        'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}
    
    def _init_transforms(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.img_transform = transforms.Compose([
            transforms.Resize(224,Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def get_sample(self, index):
        sample = self.data[index]
        image = os.path.join(self.image_base_path, sample['image'])
        audio = os.path.join(self.audio_base_path, sample['wav'])
        return image, audio
    
    def check_sample_in_local(self, index):
        img, aud = self.get_sample(index)
        img_local = os.path.join(self.local_dir + "/frame", img.split('/')[-1])
        aud_local = os.path.join(self.local_dir + "/audio", aud.split('/')[-1])
        return os.path.exists(img_local), os.path.exists(aud_local)
    
    def download_sample(self, index):

        img_rem_path, aud_rem_path = self.get_sample(index)
        if all(self.check_sample_in_local(index)):
            print("Sample already downloaded")
        else:
            print(f"Sample NOT in {self.local_dir} \n-> download  ")
            command = ["scp", "-P", "2122", "asantos@pirineus3.csuc.cat:" + img_rem_path, self.local_dir+"/frame"]
            print(command)
            subprocess.run(command)
            command = ["scp", "-P", "2122", "asantos@pirineus3.csuc.cat:" + aud_rem_path, self.local_dir+"/audio"]
            print(command)
            subprocess.run(command)
            
        img_local_path = os.path.join(self.local_dir+"/frame", img_rem_path.split('/')[-1])
        aud_local_path = os.path.join(self.local_dir+"/audio", aud_rem_path.split('/')[-1])

        return img_local_path, aud_local_path
   
    def load_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = self.img_transform(img)
        return img
    
    def show_image(self, img_path):
        img = cv2.imread(img_path)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_audio_to_spec(self, audio_path, processing_type="DAVENet",r_nFrames=False):
        if processing_type == "DAVENet":
            audio = self.load_audio_DAVENet(audio_path,r_nFrames)
        elif processing_type == "SSL-TIE":
            audio = self.load_audio_SSLTIE(audio_path)
        else:
            raise ValueError(f"Unknown processing type: {processing_type}")
        return audio
    
    def load_audio_SSLTIE(self, audio_path):
        samples, samplerate = torchaudio.load(audio_path)

        if samples.shape[1] < samplerate * 10:
            n = int(samplerate * 10 / samples.shape[1]) + 1
            samples = samples.repeat(1, n)

        samples = samples[...,:samplerate*10]

        spectrogram  =  audio_T.MelSpectrogram(
                sample_rate=samplerate,
                n_fft=512,
                hop_length=239, 
                n_mels=257,
                normalized=True
            )(samples)
        spectrogram =  self.AmplitudeToDB(spectrogram)
        return spectrogram
    
    def preemphasis(self,signal,coeff=0.97):
        """perform preemphasis on the input signal.
        
        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
        :returns: the filtered signal.
        """    
        return np.append(signal[0],signal[1:]-coeff*signal[:-1])
    
    def load_audio_DAVENet(self, file,r_nFrames):
        audio_type = self.audio_conf.get('audio_type', 'melspectrogram')
        if audio_type not in ['melspectrogram', 'spectrogram']:
            raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        preemph_coef = self.audio_conf.get('preemph_coef', 0.97)
        sample_rate = self.audio_conf.get('sample_rate', 16000)
        window_size = self.audio_conf.get('window_size', 0.025)
        window_stride = self.audio_conf.get('window_stride', 0.01)
        window_type = self.audio_conf.get('window_type', 'hamming')
        num_mel_bins = self.audio_conf.get('num_mel_bins', 40)
        target_length = self.audio_conf.get('target_length', 2048)
        use_raw_length = self.audio_conf.get('use_raw_length', False)
        padval = self.audio_conf.get('padval', 0)
        fmin = self.audio_conf.get('fmin', 20)
        n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        y, sr = librosa.load(file,sr=sample_rate)
        
        if y.size == 0:
            y = np.zeros(200)
        y = y - y.mean()
        y = self.preemphasis(y, preemph_coef)
        # compute mel spectrogram
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length,
            window=self.windows.get(window_type, self.windows['hamming']))
        spec = np.abs(stft)**2
        if audio_type == 'melspectrogram':
            mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
            melspec = np.dot(mel_basis, spec)
            logspec = librosa.power_to_db(melspec, ref=np.max)
        elif audio_type == 'spectrogram':
            logspec = librosa.power_to_db(spec, ref=np.max)
        n_frames = logspec.shape[1]
        if use_raw_length:
            target_length = n_frames
        p = target_length - n_frames
        if p > 0:
            logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
                constant_values=(padval,padval))
        elif p < 0:
            logspec = logspec[:,0:p]
            n_frames = target_length
        logspec = torch.FloatTensor(logspec)
        if r_nFrames:
            return logspec.unsqueeze(0), n_frames
        return logspec.unsqueeze(0)
   
class MatchmapVideoGenerator(ABC):
    def __init__(self,device,img, spec, matchmap):
        super().__init__()
        self.device = device
        self.device = device
        self.spec = Variable(spec.unsqueeze(0)).to(device, non_blocking=True) if spec.dim() == 3 else Variable(spec).to(device, non_blocking=True)
        self.image = Variable(img.unsqueeze(0)).to(device, non_blocking=True) if img.dim() == 3 else Variable(img).to(device, non_blocking=True)
        self.matchmap = matchmap
    
    @abstractmethod
    def compute_matchmap(self):
        pass
    
    def normalize_img(self, value, vmax=None, vmin=None):
        '''
        Normalize heatmap
        '''
        vmin = value.min() if vmin is None else vmin
        vmax = value.max() if vmax is None else vmax
        if not (vmax - vmin) == 0:
            value = (value - vmin) / (vmax - vmin)  # vmin..vmax
        return value
    
    def get_frame_match(self, img_np, matchmap_np, frame_idx):
        assert img_np.ndim == 3, "img_np should be a 3D numpy array"
        assert matchmap_np.ndim == 3, "matchmap_np should be a 3D numpy array"

        matchmap_i = matchmap_np[frame_idx]
        matchmap_i = cv2.resize(matchmap_i, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
        matchmap_i = self.normalize_img(matchmap_i)
        matchmap_i_photo = (matchmap_i * 255).astype(np.uint8)
        matchmap_i_photo = cv2.applyColorMap(matchmap_i_photo, cv2.COLORMAP_JET)
        matchmap_i_photo = cv2.addWeighted(matchmap_i_photo, 0.5, img_np, 0.5, 0)
        return matchmap_i_photo
    
    def create_video_f(self,img_np, matchmap_np, output_path="matchmap_video.mp4", fps=1):
        n_frames = matchmap_np.shape[0]
        
        # Make sure img_np is in uint8 format
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Make sure img_np has correct dimensions (224, 224, 3)
        if img_np.shape[:2] != (224, 224):
            img_np = cv2.resize(img_np, (224, 224))
        
        # Use proper codec for compatibility
        # For better compatibility, try 'avc1' or 'H264' instead of 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        
        # Alternative codec options if 'avc1' doesn't work:
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # More compatible but lower quality
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (224, 224))
        
        if not out.isOpened():
            print("Failed to create VideoWriter. Trying alternative codec...")
            # Try with different codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path.replace('.mp4', '.avi'), fourcc, fps, (224, 224))
        
        for i in range(n_frames):
            frame = self.get_frame_match(img_np, matchmap_np, i)
            
            # Ensure frame is the correct format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
                
            # Ensure frame has the right shape
            if frame.shape[:2] != (224, 224):
                frame = cv2.resize(frame, (224, 224))
                
            # Verify frame is BGR (OpenCV's default format)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                out.write(frame)
            else:
                print(f"Warning: Frame {i} has incorrect format. Shape: {frame.shape}")
        
        out.release()
        print(f"Video created at: {output_path}")
        
        # Verify the file was created and has a non-zero size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Success! Video file created: {os.path.getsize(output_path)} bytes")
        else:
            print("Error: Video file was not created properly")
    
    def create_video(self,output_path):
        img_np = self.image[0].cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = self.normalize_img(img_np)
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        if self.matchmap is None:
            self.matchmap = self.compute_matchmap()

        matchmap_np = self.matchmap.cpu().numpy()
        n_frames = matchmap_np.shape[0]
        self.create_video_f(img_np, matchmap_np, output_path, fps=n_frames/20.48) # 10 sec duration

    def add_audio_to_video(self, video_path, audio_path):
        """
        Add audio to video using ffmpeg and overwrite the output file if it exists.
        """
        temp_output = "temp_output.mp4"
        
        # Ensure the audio file exists
        if not os.path.exists(audio_path):
            raise Exception("Error: Audio file not found.")

        # Use ffmpeg to merge audio and video
        command = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i", video_path,  # Input video
            # "-stream_loop", "-1",  # Infinite loop for audio
            "-i", audio_path,  # Input audio
            "-map", "0:v",  # Video stream from first input
            "-map", "1:a",  # Audio stream from second input
            "-c:v", "copy",  # Copy video codec (no re-encoding)
            "-c:a", "libmp3lame",  # Encode audio in mp3 format
            "-t", "20.48",  # 20.48 sec duration
            temp_output
        ]
        
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            if os.path.exists(temp_output):  # Ensure the temporary file exists
                os.remove(video_path)  # Delete the original video file
                shutil.move(temp_output, video_path)  # Rename the temporary file
            else:
                raise Exception("Error: Temporary output file not created.")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error during ffmpeg execution: {e.stderr.decode('utf-8')}")

    def create_video_with_audio(self, output_path, audio_path):
        self.create_video(output_path)
        self.add_audio_to_video(output_path, audio_path)

    

class MatchmapVideoGeneratorSSL_TIE(MatchmapVideoGenerator):
    def __init__(self,model, device, img, spec,args, matchmap = None):
        super().__init__(device,img,spec,matchmap)
        self.model = model
        self.args = args

    def compute_matchmap(self):
        self.model.eval()
        with torch.no_grad():
            img_emb,aud_emb = self.model(self.image.float(), self.spec.float(), self.args)
            img_emb = img_emb.squeeze(0)
            aud_emb = aud_emb.squeeze(0)
            self.matchmap = torch.einsum('ct, chw -> thw',aud_emb,img_emb)
        return self.matchmap
    
    
    def create_video_f(self,img_np, matchmap_np, output_path="matchmap_video.mp4", fps=1):
        n_frames = matchmap_np.shape[0]
        
        # Make sure img_np is in uint8 format
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Make sure img_np has correct dimensions (224, 224, 3)
        if img_np.shape[:2] != (224, 224):
            img_np = cv2.resize(img_np, (224, 224))
        
        # Use proper codec for compatibility
        # For better compatibility, try 'avc1' or 'H264' instead of 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        
        # Alternative codec options if 'avc1' doesn't work:
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # More compatible but lower quality
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (224, 224))
        
        if not out.isOpened():
            print("Failed to create VideoWriter. Trying alternative codec...")
            # Try with different codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path.replace('.mp4', '.avi'), fourcc, fps, (224, 224))
        
        for i in range(n_frames):
            frame = self.get_frame_match(img_np, matchmap_np, i)
            
            # Ensure frame is the correct format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
                
            # Ensure frame has the right shape
            if frame.shape[:2] != (224, 224):
                frame = cv2.resize(frame, (224, 224))
                
            # Verify frame is BGR (OpenCV's default format)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                out.write(frame)
            else:
                print(f"Warning: Frame {i} has incorrect format. Shape: {frame.shape}")
        
        out.release()
        print(f"Video created at: {output_path}")
        
        # Verify the file was created and has a non-zero size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Success! Video file created: {os.path.getsize(output_path)} bytes")
        else:
            print("Error: Video file was not created properly")
            


def load_model(ckpt_path, args):
    model = AVENet(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model, device

def update_json_file(path_2_json, epochs_dict, epoch, key, link_2_store):
    """
    Updates the JSON file with a new video link for the given epoch.

    Args:
        path_2_json (str): Path to the JSON file to update.
        epochs_dict (dict): The dictionary containing epoch data.
        epoch (int): The epoch number to update.
        key (str): The key for the dict ["weights_link" | "video_link"]
        link_2_store (str): The video link to add.
    Returns 
        epochs_dict (dict): the updated dict
    """
    try:
        # Ensure the epoch exists in the dictionary
        if str(epoch) not in epochs_dict:
            epochs_dict[str(epoch)] = {}

        # Update the link for the epoch
        epochs_dict[str(epoch)][key] = link_2_store

        # Write the updated dictionary back to the JSON file
        with open(path_2_json, "w", encoding="utf-8") as json_file:
            json.dump(epochs_dict, json_file, indent=4)

        return epochs_dict
    
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error updating JSON file: {e}")



#FOR INTERACTING WITH THE CLUSTER

def folder_exists_in_cluster(remote_path):
    # Command to check if the folder exists
    check_command = f"ssh -p 2122 asantos@pirineus3.csuc.cat '[ -d \"{remote_path}\" ] && echo Exists || echo NotExists'"
    result = subprocess.run(check_command, shell=True, capture_output=True, text=True)
    
    return result.stdout.strip() == "Exists"

def count_files_in_folder(remote_path):
    # Command to count files in the folder
    count_command = f"ssh -p 2122 asantos@pirineus3.csuc.cat 'ls -1 \"{remote_path}\" | wc -l'"
    result = subprocess.run(count_command, shell=True, capture_output=True, text=True)

    return int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0

def list_files_in_remote_folder(remote_path):
    # Command to list all files in the remote folder
    list_command = f"ssh -p 2122 asantos@pirineus3.csuc.cat 'ls -1 \"{remote_path}\"'"
    result = subprocess.run(list_command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    else:
        print(f"Error listing files in remote folder: {result.stderr}")
        return []

def download_remote_file(remote_path, local_path):
    """
    Downloads a specific file from a remote folder to a local folder using scp.
    """
    # Check if the file already exists locally
    if os.path.exists(local_path):
        print(f"File already exists locally: {local_path}")
        return

    # Ensure the local directory exists
    local_dir = os.path.dirname(local_path)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir) 
    

    # Command to download the specific file
    download_command = f"scp -P 2122 asantos@pirineus3.csuc.cat:\"{remote_path}\" \"{local_path}\""
    result = subprocess.run(download_command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Successfully downloaded {remote_path} to {local_path}")
    else:
        print(f"Error downloading {remote_path}: {result.stderr}")
        raise FileNotFoundError()


def upload_file_to_cluster(local_path, remote_path):
    """
    Uploads a specific file from a local folder to a remote folder using scp.
    """
    # Check if the local file exists
    if not os.path.exists(local_path):
        print(f"Local file does not exist: {local_path}")
        return

    # Ensure the remote directory exists
    remote_dir = os.path.dirname(remote_path)
    create_remote_dir_command = f"ssh -p 2122 asantos@pirineus3.csuc.cat 'mkdir -p \"{remote_dir}\"'"
    subprocess.run(create_remote_dir_command, shell=True, capture_output=True, text=True)

    # Command to upload the specific file
    upload_command = f"scp -P 2122 \"{local_path}\" asantos@pirineus3.csuc.cat:\"{remote_path}\""
    result = subprocess.run(upload_command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Successfully uploaded {local_path} to {remote_path}")
    else:
        print(f"Error uploading {local_path}: {result.stderr}")


# To Download the weights of a model with its json
class ModelOutputRetriever:
    def __init__(self,model_name,local_dir):
        self.model_name = model_name
        self.model_dir = os.path.join("models",model_name)
        self.local_dir = local_dir
        #Check if the folder exists
        if not folder_exists_in_cluster(self.model_dir):
            raise Exception(f"Model {model_name} doesn't exist among models")
        
        # - if more than one ask the user which one if not use that one
        filenames = list_files_in_remote_folder(self.model_dir)
        if len(filenames) > 1:
            request_string = f"There's more than one json for {model_name}\nChoose one:\n"
            for idx, flname in enumerate(filenames):
                file_string = f"[{str(idx)}] {flname}\n"
                request_string = request_string + file_string
            idx_to_process = int(input(request_string))

            print(f"Using [{str(idx_to_process)}] {filenames[idx_to_process]}")

        else:
            idx_to_process = 0
            print(f"Using {filenames[0]}")
        
        #check if json is in local if not download
        json_file = filenames[idx_to_process]
        if not self.check_in_local(json_file):
            download_remote_file(
                remote_path = os.path.join(self.model_dir,json_file),
                local_path = os.path.join(self.local_dir,json_file)
                ) 
        self.json_file = os.path.join(self.local_dir,json_file)
        
        with open(self.json_file, 'r') as f:
            self.data_json = json.load(f)

        self.check_if_available()
        

    def check_in_local(self,file):
        return os.path.exists(os.path.join(self.local_dir,file))
            

    def get_number_of_epochs(self):
        return len(self.data_json) - 1
    
    def get_parameters(self):
        return self.data_json['parameters']
    
    def get_weigths_link(self, epoch):
        return self.data_json[str(epoch)]['weights_link']
    
    def download_weigths(self, epoch):
        weigths_link = self.get_weigths_link(epoch)
        weigths_name = self.model_name + "-" + weigths_link.split('/')[-1]
        if not self.check_in_local(weigths_name):
            print("Downloading weigths")
            download_remote_file(
                remote_path = weigths_link,
                local_path = os.path.join(self.local_dir,weigths_name)
            )
        else:
            print("Weights already in local dir")
        return os.path.join(self.local_dir,weigths_name)
    
    def get_video_link(self, epoch):
        return self.data_json[str(epoch)]['video_link']
    
    def download_video(self, epoch, path_2_save = None):
        video_link = self.get_video_link(epoch)
        sample_idx = self.data_json["parameters"]["val_video_idx"]
        video_name = f'mm_{self.model_name}-epoch{epoch}_val_{sample_idx}.mp4'
        if not self.check_in_local(video_name):
            print("Downloading video")
            download_remote_file(
                remote_path = video_link,
                local_path = os.path.join(path_2_save if path_2_save else self.local_dir
                                          ,video_name)
            )
        else:
            print("Video already in local dir")
        return os.path.join(self.local_dir,video_name)
    
    def check_if_available(self):
        """
        Check if 7 seven days have passed since the time creation
        """
        # Parse the time_creation string into a datetime object
        creation_time = datetime.strptime(self.data_json["parameters"]["time_creation"], "%Y-%m-%d %H:%M")

        # Check if 7 days have passed since the creation time
        if datetime.now() - creation_time > timedelta(days=7):
            print("The model is no longer available.")
            return False
        else:
            print("The model is still available. Creation time: ",creation_time)
            time_difference = timedelta(days=7) - (datetime.now() - creation_time)
            days = time_difference.days
            hours = time_difference.seconds // 3600

            if days > 0:
                print(f"The model will be available for {days} more days.")
            else:
                print(f"The model will be available for {hours} more hours.")
            return True
    
def inference_maker(model_name, epoch, split, sample_idx, local_dir_saving):
    inference_name = f'mm_{model_name}-epoch{epoch}_{split}_{sample_idx}.mp4'
    inference_local_path = os.path.join(local_dir_saving, "inferences", str(sample_idx), inference_name)

    if os.path.exists(inference_local_path):
        print("Inference already in local")
        return inference_local_path
    print("Proceed to make the inference")

    # Ensure the directory for inference_local_path exists
    inference_dir = os.path.dirname(inference_local_path)
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    model_weights_path = os.path.join(local_dir_saving,
                                      f'{model_name}-epoch{epoch}.pth.tar')
    
    # Simulate command-line arguments for loading the model
    sys.argv = ['script_name', '--order_3_tensor', '--simtype', 'MISA']

    args = get_arguments()

    #Check if the model exist in local
    if os.path.exists(model_weights_path):
        print("The model is in local")
        model, device = load_model(model_weights_path,args)
        print("-Model Loaded")
    else:
        #Request if user wants to download the model
        user_input = input("Model weights not found locally. Do you want to download them? (yes/no): ").strip().lower()
        if user_input == "yes":
            mor = ModelOutputRetriever(model_name, local_dir_saving)
            model_weights_path = mor.download_weigths(epoch)
            model, device = load_model(model_weights_path, args)
            print("-Model Loaded")
        else:
            raise FileNotFoundError("Model weights are required but not available locally.")
    
    #Check if the audio and the image of the sample index is in local
    json_file = f'garbage/{split}.json'
    if os.path.exists(json_file):
        gs = GetSampleFromJson(json_file, local_dir_saving)
        img_local_path, aud_local_path = gs.download_sample(sample_idx)
    else:
        raise FileNotFoundError("json file not found for retrieving the samples")

    img = gs.load_image(img_local_path)
    spec = gs.load_audio_to_spec(aud_local_path)

    mgv = MatchmapVideoGeneratorSSL_TIE(model, device, img, spec, args)
    mgv.create_video_with_audio(inference_local_path, aud_local_path)
    
    
    return inference_local_path