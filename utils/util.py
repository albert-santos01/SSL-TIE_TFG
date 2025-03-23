import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import wandb
from PIL import Image

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
    Its the dot product.
    """
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    # D = I.size(0)
    # H = I.size(1)
    # W = I.size(2)
    # T = A.size(1)                                                                                                                     
    # Ir = I.view(D, -1).t()
    # matchmap = torch.mm(Ir, A)
    # matchmap = matchmap.view(H, W, T)  
    matchmap = torch.einsum('hwc,tc->hwt', I, A)
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
def volumemap_sim(volume_matrix, simtype='MISA'):
    """
    Computes the similarity scoring for all the combinations B x B
        volume_matrix (B x B x T x H x W)
        simtype MISA | SISA | SIMA
        Returns (B x B) similarity matrix
    """
    assert(volume_matrix.dim() == 5)

    if simtype== 'MISA':
        #MI
       volume_matrix, _ =volume_matrix.max(-1) # x spatial
       volume_matrix, _ =volume_matrix.max(-1) # y spatial
        #MISA
       volume_matrix =volume_matrix.mean(-1) # t temporal
    elif simtype == 'SISA':
        volume_matrix = volume_matrix.mean(-1)
        volume_matrix = volume_matrix.mean(-1)
        volume_matrix = volume_matrix.mean(-1)
    elif simtype == 'SIMA':
        volume_matrix,_ = volume_matrix.max(2) # MA
        volume_matrix = volume_matrix.mean(-1)
        volume_matrix = volume_matrix.mean(-1) #SIMA
    else:
        raise ValueError('Unknown similarity type: %s' % simtype)

    assert(volume_matrix.dim() == 3)
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


    

def infoNCE_loss(image_outputs, audio_outputs,args):
    """
        images_outputs (B x C x H x W) 
        audio_outputs  (B x C x T)
        Assumption: the channel dimension is already normalized
        Returns the InfoNCE loss
    """
   
    B = image_outputs.size(0)
    mask = torch.eye(B, device=image_outputs.device)

    sims =  similarity_matrix_bxb(image_outputs, audio_outputs,args.temperature,args.simtype)
    pos = sims * mask
    neg = sims * (1 - mask)

    # This iterates the images against their negative audios...
    loss_v_a = -torch.log(pos.sum(dim=1) / (pos.sum(dim=1) + neg.sum(dim=1))).mean() / 2 
    # This iterates the audios against their negative images...
    loss_a_v = -torch.log(pos.sum(dim=0) / (pos.sum(dim=0) + neg.sum(dim=0))).mean() / 2
    loss = loss_v_a + loss_a_v

    return loss











    