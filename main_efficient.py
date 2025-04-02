import os
import socket
import sys
import warnings
import gc

# Ignore all warnings of any type
warnings.filterwarnings("ignore")

# print("Silencing warnings from modules...")
# Save original stderr
# original_stderr = sys.stderr

# Redirect stderr to a "black hole" (null device)
# sys.stderr = open(os.devnull, 'w')

# Import modules (silently)
import wandb

from kornia.geometry.transform.affwarp import scale
from numpy.lib.type_check import imag
import torch
from torch.nn.modules import loss
from torch.optim import *
from torch.serialization import save
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable, gradcheck
from torch.utils.data import Dataset, DataLoader, dataset
from tensorboardX import SummaryWriter

import torch.nn.functional as F
import numpy as np
import random
import json
import ipdb
import time
import cv2
import kornia
import sklearn
from PIL import Image

from models.model import AVENet

from datasets import GetAudioVideoDataset, PlacesAudio
from opts import get_arguments
from utils.utils import AverageMeter, reverseTransform, accuracy
import xml.etree.ElementTree as ET
from utils.eval_ import Evaluator
from sklearn.metrics import auc
from tqdm import tqdm

from utils.util import topk_accuracy, update_json_file, MatchmapVideoGenerator,\
      vis_loader, prepare_device, vis_heatmap_bbox, tensor2img, \
      sampled_margin_rank_loss, computeMatchmap, vis_matchmap, infoNCE_loss
from utils.tf_equivariance_loss import TfEquivarianceLoss
import multiprocessing
from datetime import datetime


from utils.utils import save_checkpoint, AverageMeter,  \
     Logger, neq_load_customized, ProgressMeter


# Restore original stderr
# sys.stderr.close()
# sys.stderr = original_stderr

# print("Back to normal logging!")


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value


def cal_auc(iou):
    results = []
    for i in range(21):
        result = np.sum(np.array(iou) >= 0.05 * i)
        result = result / len(iou)
        results.append(result)
    x = [0.05 * i for i in range(21)]
    auc_ = auc(x, results)

    return auc_



def set_path(args):
    if args.resume: 
        exp_path = os.path.dirname(os.path.dirname(args.resume)) # It should be the parent directory of the model
        links_path = ""
    elif args.test: 
        exp_path = os.path.dirname(os.path.dirname(args.test))
        links_path = ""
    else:
        # exp_path = 'ckpts/{args.exp_name}'.format(args=args)
                                
        # Check if we are using the cluster 
        if multiprocessing.cpu_count() > 16: 
            exp_path = os.path.expandvars('$SCRATCH/{args.exp_name}'.format(args=args))
        else:
            exp_path = 'garbage/{args.exp_name}'.format(args=args)

        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        else:
            print('The experiment folder already exists')
        
        links_path = os.path.expandvars('$HOME/models/{args.exp_name}'.format(args=args))
        
        if not os.path.exists(links_path):
            os.makedirs(links_path)
        
        #Reuse variable
        links_path = os.path.join(links_path, 'links_{args.exp_name}_{args.job_id}.json'.format(args=args))
        
        # Load existing data if JSON file exists, else start with an empty dictionary
        if os.path.exists(links_path):
            with open(links_path, "r") as json_file:
                args.epochs_data = json.load(json_file)
        else:
            
            # Create dictionary with epoch entries
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            epochs_data = {
                "parameters": {
                    "time_creation": current_time,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "simtype":  args.simtype,
                    "temperature": args.temperature,
                    "val_video_idx": args.val_video_idx
                }
            }
            args.epochs_data = epochs_data


        

    img_path = os.path.join(exp_path, 'img') 
    model_path = os.path.join(exp_path, 'model') 

    if not os.path.exists(img_path): 
        os.makedirs(img_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)
    return img_path, model_path, exp_path, links_path


def train_one_epoch(train_loader, model, criterion, optim, device, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    losses_cl = AverageMeter('Loss',':.4f')
    losses_cl_ts = AverageMeter('Loss',':.4f')
    losses_ts = AverageMeter('Loss',':.4f')

    progress = ProgressMeter(                             
        len(train_loader),
        # [batch_time, data_time, losses, top1_meter, top5_meter],
        [batch_time, data_time, losses],
        prefix='Epoch:[{}]'.format(epoch))
    
    # basicblock = BasicBlock()
    model.train()
    

    
    end = time.time()
    tic = time.time()

    lambda_trans_equiv = args.trans_equi_weight
    
    loss_debug = 0

    for idx, (image, spec, audio, name, img_numpy) in enumerate(train_loader):
        data_time.update(time.time() - end)
        spec = Variable(spec).to(device, non_blocking=True)
        image = Variable(image).to(device, non_blocking=True) 
        
        B = image.size(0)
        torch.autograd.set_detect_anomaly(True)
        # First branch of the siamese network
        imgs_out, auds_out = model(image.float(), spec.float(), args, mode='train')
        
        # loss_cl =  sampled_margin_rank_loss(imgs_out, auds_out, margin=1., simtype=args.simtype)   
        loss_cl = infoNCE_loss(imgs_out,auds_out, args)        

        if args.siamese:

            match_map_b = torch.einsum('bhwd,btd -> bhwt',imgs_out, auds_out)

            tf_equiv_loss = TfEquivarianceLoss(
                            transform_type='rotation',
                            consistency_type=args.equi_loss_type,
                            batch_size=B,
                            max_angle=args.max_rotation_angle,
                            input_hw=(224, 224),
                            )
            tf_equiv_loss.set_tf_matrices()

            transformed_image = tf_equiv_loss.transform(image)

            # Second branch of the siamese network
            imgs_out_ts, auds_out_ts = model(transformed_image.float(), spec.float(), args, mode='train')
            loss_cl_ts = sampled_margin_rank_loss(imgs_out_ts, auds_out_ts, margin=1., simtype=args.simtype)
            # top1_ts, top5_ts = calc_topk_accuracy(out_ts, target, (1,5))

            match_map_b_ts = torch.einsum('bhwd,btd -> bhwt',imgs_out_ts, auds_out_ts)

            ts_match_map = tf_equiv_loss.transform(match_map_b) # TODO: Modify for 3rd order tensor
            loss_ts = tf_equiv_loss(match_map_b_ts, ts_match_map) # This is the transformation equivariance loss "Siamese network"
            loss = 0.5*(loss_cl + loss_cl_ts) + lambda_trans_equiv * loss_ts 

            #Log batch metrics to  wandb
            if args.use_wandb:
                wandb.log({ "train_loss_step": loss.item(), "train_loss_cl_step": loss_cl.item(),
                            "train_loss_cl_ts_step": loss_cl_ts.item(), 
                            "train_loss_ts_step": loss_ts.item(), "step": wandb.run.step})
        else:
            
            loss = loss_cl
            loss_debug += loss.item()

            #Log batch metrics to  wandb
            if args.use_wandb:
                wandb.log({ "train_loss_step": loss.item(), "step": wandb.run.step})

        losses.update(loss.item(), B)
        losses_cl.update(loss_cl.item(), B)
        if args.siamese:
            losses_cl_ts.update(loss_cl_ts.item(), B) 
            losses_ts.update(loss_ts.item(), B) 


        

        
        
        optim.zero_grad()
        loss.backward()
        # for name, param in model.named_parameters(): 
        #     if param.grad is None:
        #         print(f"Parameter {name} has no gradient.---------------------")
        #     else: 
        #         print(f"the following {name} has parameter")

        # raise Exception("STOP BITCH")
        optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)
        
        args.iteration += 1
                

        if args.mem_efficient:
            if idx % args.free_mem_freq == 0:
                torch.cuda.empty_cache()
                

    wandb.log({"T-epoch": time.time()-tic, "epoch": epoch}) if args.use_wandb else None
    print('Epoch: [{0}][{1}/{2}]\t'
        'T-epoch:{t:.2f}\t'.format(epoch, idx, len(train_loader), t=time.time()-tic))

    
    # Log the epoch metrics to # wandb
    if args.use_wandb:
        if args.siamese:
            wandb.log({
                        "train_loss_epoch": losses.avg,
                        "train_loss_cl_epoch": losses_cl.avg, "train_loss_cl_ts_epoch": losses_cl_ts.avg,
                        "train_loss_ts_epoch": losses_ts.avg, "epoch": epoch})
        else:
            assert losses.avg == loss_debug/len(train_loader), "Losses are not equal !"
            wandb.log({"train_loss_epoch": losses.avg, "epoch": epoch})
        
    args.train_logger.log('train Epoch: [{0}][{1}/{2}]\t'
                    'T-epoch:{t:.2f}\t'.format(epoch, idx, len(train_loader), t=time.time()-tic))
    
    if args.mem_efficient:
        gc.collect()
        torch.cuda.empty_cache()

    return losses.avg, 0

    


def validate(val_loader, model, criterion, device, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_a_v_meter = AverageMeter()
    acc_v_a_meter = AverageMeter() 
    

    tic = time.time()
    save_dir = os.path.join(args.img_path, "val_imgs", str(epoch)) 

    model.eval()
    
    video_gen = False
    with torch.no_grad():
        end = time.time()
        for idx, (image, spec, audio_path, name, im) in tqdm(enumerate(val_loader), total=len(val_loader)):

            spec = Variable(spec).to(device, non_blocking=True)
            image = Variable(image).to(device, non_blocking=True)
            B = image.size(0)
            # vis_loader(image, spec,idx)

            imgs_out, auds_out = model(image.float(), spec.float(), args, mode='val')
                            
            loss_cl,sims = infoNCE_loss(imgs_out,auds_out, args,return_S=True)
            acc_v_a, acc_a_v = topk_accuracy(sims,k=5)
            

            losses.update(loss_cl.item(), B)
            acc_a_v_meter.update(acc_a_v)
            acc_v_a_meter.update(acc_v_a)
           
            batch_time.update(time.time() - end)
            end = time.time()
            

            if args.video and (epoch % args.val_video_freq) == 0:
                if not video_gen:
                    if B*idx <= args.val_video_idx < idx*B + B:
                        idx_B   =  args.val_video_idx % B 
                        img_emb = imgs_out[idx_B]
                        aud_emb = auds_out[idx_B]
                        audio = audio_path[idx_B]
                        

                        matchmap = computeMatchmap(img_emb,aud_emb)
                        frame   = image[idx_B]

                        mgv = MatchmapVideoGenerator(model,device,frame,spec[idx_B],args,matchmap)

                        video_dir = os.path.join(args.img_path,"val_videos")
                        if not os.path.exists(video_dir):       
                            os.makedirs(video_dir)
                        video_dir = os.path.join(video_dir, f"epoch_{epoch}.mp4")

                        mgv.create_video_with_audio(video_dir,audio) 
                        video_gen = True

                        #Save link to the json file
                        args.epochs_data = update_json_file(args.links_path, args.epochs_data, epoch, "video_link", video_dir)

                        if args.use_wandb:
                            wandb.log({"val_video": wandb.Video(video_dir, caption=f"Epoch {epoch}"), "epoch": epoch})



    """    
        recalls = calc_recalls(image_output, audio_output, nframes, simtype=args.simtype)
        A_r10 = recalls['A_r10']
        I_r10 = recalls['I_r10']
        A_r5 = recalls['A_r5']
        I_r5 = recalls['I_r5']
        A_r1 = recalls['A_r1']
        I_r1 = recalls['I_r1']

    print(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
          .format(A_r10=A_r10, I_r10=I_r10, N=N_examples), flush=True)
    print(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
          .format(A_r5=A_r5, I_r5=I_r5, N=N_examples), flush=True)
    print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
          .format(A_r1=A_r1, I_r1=I_r1, N=N_examples), flush=True)

    return recalls
       """             
    
    if args.use_wandb:
        wandb.log({
            "val_loss": losses.avg,
            "epoch": epoch,
            "A->V R@5": acc_a_v_meter.avg,
            "V->A R@5": acc_v_a_meter.avg
        })
    print('Epoch: [{0}]\t Eval '
          'Loss: {loss.avg:.4f}  \t T-epoch: {t:.2f} \t'
          'A->V R@5: {a_v_r5:.4f} \t V->A R@5: {v_a_r5:.4f}'
          .format(epoch, loss=losses, t=time.time()-tic, 
                  a_v_r5=acc_a_v_meter.avg, v_a_r5=acc_v_a_meter.avg))
    return losses.avg, acc_a_v_meter.avg, acc_v_a_meter.avg

        

def test(test_loader, model, criterion, device, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')

    # Compute ciou
    val_ious_meter = []

    # dir for saving validationset heatmap images 
    save_dir = os.path.join(args.img_path, "test_imgs", str(epoch), args.test_set) 

    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (image, spec, audio, name, im) in tqdm(enumerate(test_loader), total=len(test_loader)):         
            spec = Variable(spec).to(device)
            image = Variable(image).to(device)
            B = image.size(0)

            heatmap, out, Pos, Neg, out_ref = model(image.float(), spec.float(), args, mode='val')
            target = torch.zeros(out.shape[0]).to(device, non_blocking=True).long()
            loss =  criterion(out, target)
            losses.update(loss.item(), B)
            batch_time.update(time.time() - end)
            end = time.time()

            heatmap_arr =  heatmap.data.cpu().numpy()


            for i in range(spec.shape[0]):
                
                heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                heatmap_now = normalize_img(-heatmap_now)
                gt_map = np.zeros([224,224])
                bboxs = []

                if (not args.dataset_mode=='Flickr') and (args.test_set == 'VGGSS'):
                    gt = ET.parse(args.vggss_test_path + '/anno/' + '%s.xml' % name[i]).getroot()
                    
                    for child in gt:                 
                        if child.tag == 'bbox':
                            for childs in child:
                                bbox_normalized = [ float(x.text) for x in childs  ]
                                bbox = [int(x*224) for x in bbox_normalized ]           
                                bboxs.append(bbox)
                
                    for item in bboxs:
                        xmin, ymin, xmax, ymax = item
                        gt_map[ymin:ymax, xmin:xmax] = 1

                elif (args.dataset_mode=='Flickr') or (args.test_set =='Flickr'):
                    gt = ET.parse(args.soundnet_test_path + '/anno/' + '%s.xml' % name[i]).getroot()

                    for child in gt: 
                        for childs in child:
                            bbox = []
                            if childs.tag == 'bbox':
                                for index,ch in enumerate(childs):
                                    if index == 0:
                                        continue
                                    bbox.append(int(224 * int(ch.text)/256))
                            bboxs.append(bbox)  

                    for item_ in bboxs:
                        temp = np.zeros([224,224])
                        (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
                        temp[item_[1]:item_[3],item_[0]:item_[2]] = 1
                        gt_map += temp
                    gt_map /= 2         
                    gt_map[gt_map>1] = 1
                    

                else:
                    print('Testing dataset Not Assigned !')

                pred =  heatmap_now
                pred = 1 - pred
                threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]    # 计算threshold
                pred[pred>threshold]  = 1
                pred[pred<1] = 0
                evaluator = Evaluator()
                ciou, inter, union = evaluator.cal_CIOU(pred, gt_map, 0.5)

                val_ious_meter.append(ciou)  

                heatmap_vis = np.expand_dims(heatmap_arr[i], axis=0)
                # img_vis = img_arrs[i]
                img_vis_tensor = image[i]
                img_vis = tensor2img(img_vis_tensor.data.cpu())

                name_vis = name[i]
                bbox_vis = bboxs
    
                
                heatmap_img = vis_heatmap_bbox(heatmap_vis, img_vis, name_vis,\
                        bbox=bbox_vis, ciou=ciou, save_dir=save_dir )
            
    mean_ciou = np.sum(np.array(val_ious_meter) >= 0.5)/ len(val_ious_meter)    
    auc_val = cal_auc(val_ious_meter)

            
    print('Test: \t Epoch: [{0}]\t'
          'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f} MeancIoU: {ciouAvg:.4f} AUC: {auc:.4f}\t'
          .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter, ciouAvg=mean_ciou, auc=auc_val))


    args.test_logger.log('Test Epoch: [{0}]\t'
                    'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f} MeancIoU: {ciouAvg:.4f} AUC:{auc:.4f} \t'
                    .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter, ciouAvg=mean_ciou, auc=auc_val))

    sys.exit(0)



def main(args):
    # To Ensure Cuda context
    print("Is cuda available",torch.cuda.is_available()) 

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))

    if args.use_cuda:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please run with flag --use_cuda False")

    args.gpus = list(range(torch.cuda.device_count()))
    print('Using GPU:', args.gpus)
    print('Number of CPUs:', multiprocessing.cpu_count())

    if args.debug_code:
        print('Debugging code')
        raise ValueError('Debugging code')
    
    # wandb initialization
    if args.use_wandb:
        config = dict(
            learning_rate = args.learning_rate,
            weight_decay = args.weight_decay,
            batch_size = args.batch_size,
            epochs = args.epochs,
            dataset_mode = args.dataset_mode,
            seed = args.seed)
        if args.run_name:
            wandb.init(project=args.project_wandb, config=config, name=args.run_name)
        else:
            wandb.init(project=args.project_wandb, config=config)
        


    
    if args.debug:
        args.n_threads=0

    # args.host_name = os.uname()[1] # For windows does not work
    args.host_name = socket.gethostname()


    print('Number of GPUs:', len(args.gpus))
    
    if torch.cuda.is_available():
        # device = torch.device('cuda:1') if len(args.gpus) > 1 else torch.device('cuda:0')
        device = torch.device('cuda:0') if len(args.gpus) > 1 else torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    best_acc = 0
    best_miou = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.img_path, args.model_path, args.exp_path, args.links_path = set_path(args)
    
    model = AVENet(args)
    model.to(device)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=args.gpus, output_device=device)
        model_without_dp = model.module
    else:
        model_without_dp = model  # Directly assign the model if not using DataParallel

    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[300,700,900], gamma=0.1) # Stupid bc it will never affect (100 epoch max)
    args.iteration = 1
    
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
            epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            
            try: 
                model_without_dp.load_state_dict(state_dict)
            except: 
                neq_load_customized(model_without_dp, state_dict, verbose=True)
        
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.test))
            epoch = 0
        print("Current execution path:", os.getcwd())
        logger_path = os.path.abspath(os.path.join(os.path.join(os.getcwd(),'img/logs/test'), os.path.dirname(args.test))) # modified
        # logger_path = os.path.join(args.img_path, 'logs', 'test')
        print(logger_path)

        if not os.path.exists(logger_path):
            os.makedirs(logger_path)


        args.test_logger = Logger(path=logger_path)
        args.test_logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
        if args.dataset_mode == 'VGGSound':
            test_dataset = GetAudioVideoDataset(args, mode='test' if args.test_set == 'VGGSS' else 'val')
        elif args.dataset_mode == 'Flickr':
            test_dataset = GetAudioVideoDataset(args, mode='test')
        elif args.dataset_mode == 'Debug':
            test_dataset = GetAudioVideoDataset(args, mode='test')

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,\
            num_workers=args.n_threads, pin_memory=True)
        
        test(test_loader, model, criterion, device, epoch, args )

    if args.placesAudio:
        train_dataset = PlacesAudio(args.placesAudio + 'train.json', args,mode='train')
        val_dataset = PlacesAudio(args.placesAudio + 'val.json', args,mode='val')

    else:
        train_dataset = GetAudioVideoDataset(args, mode='train')

    if args.dataset_mode == 'VGGSound':
        val_dataset = GetAudioVideoDataset(args, mode='test' if args.val_set == 'VGGSS' else 'val')
    elif args.dataset_mode == 'Flickr':
        val_dataset = GetAudioVideoDataset(args, mode='test')
    elif args.dataset_mode == 'Debug':
        val_dataset = GetAudioVideoDataset(args, mode='test')

    print("Creating Data loaders with %d workers" % args.n_threads)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, \
        shuffle=True, num_workers=args.n_threads, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.n_threads, drop_last=True, pin_memory=True)
    

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']+ 1
            args.iteration = checkpoint['iteration']
            best_miou = checkpoint['best_miou']
            state_dict = checkpoint['state_dict']

            try: 
                model_without_dp.load_state_dict(state_dict)
            except:
                print('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_dp, state_dict, verbose=True)
            
            print("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            
            try:
                optim.load_state_dict(checkpoint['optimizer'])
            except:
                print('[WARNING] failed to load optimizer state, initialize optimizer')
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.resume))

    else:
        print('Train the model from scratch on {0} for {1}!'.format(args.dataset_mode,args.exp_name))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    

    train_log_path = os.path.join(args.img_path, 'logs','train')
    val_log_path   = os.path.join(args.img_path, 'logs', 'val')
    
    for path in [train_log_path, val_log_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    args.train_logger = Logger(path=train_log_path)
    args.val_logger = Logger(path=val_log_path)

    args.train_logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
    
    print('\n ******************Training Args*************************')
    print('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
    print('******************Training Args*************************')

    for epoch in range(args.start_epoch, args.epochs + 1 ):
        np.random.seed(epoch)
        random.seed(epoch)
        print('Epoch: %d/%d' % (epoch, args.epochs))
        if args.mem_efficient:
            gc.collect()
            torch.cuda.empty_cache()
        start = time.time()

        train_one_epoch(train_loader, model, criterion, optim, device, epoch, args)


        print('Training time: %d seconds.' % (time.time() - start))
        
        
        
        if epoch >= args.eval_start:
            args.eval_freq = 1
            
        #It always goes here unless --eval_start x is set (default = l)
        if epoch % args.eval_freq == 0:
            #TODO: We should change this, now that it considers R@5
            val_loss, _, mean_ciou = validate(val_loader, model, criterion, device, epoch, args)

            if args.order_3_tensor: #This val_loss will be considered as the metric from now
                is_best = val_loss > best_miou               
                best_miou = max(val_loss, best_miou)                
            else:
                is_best = mean_ciou > best_miou
                best_miou = max(mean_ciou, best_miou)

            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_miou': best_miou,
                'optimizer': optim.state_dict(),
                'iteration': args.iteration}
            
            
            
            #It will save the model and also the best one if it's considered
            save_checkpoint(save_dict, is_best, 1, 
                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch), 
                keep_all=True)

        
        else:
            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_miou': best_miou,
                'optimizer': optim.state_dict(),
                'iteration': args.iteration}

            save_checkpoint(save_dict, is_best=0, gap=1, 
                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch), 
                keep_all=True)

       
        # Update the json with the new weights
        filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch)
        args.epochs_data = update_json_file(args.links_path, args.epochs_data, epoch, "weights_link", filename)

        torch.cuda.empty_cache()
        scheduler.step()
    
    print('Training from Epoch %d --> Epoch %d finished' % (args.start_epoch, args.epochs ))
    sys.exit(0)




if __name__ == "__main__":
    args=get_arguments()
    main(args)