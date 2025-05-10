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

from utils.util import  load_model, topk_accuracies, similarity_matrix_bxb, \
      topk_accuracy, update_json_file, MatchmapVideoGenerator,\
      vis_loader, prepare_device, vis_heatmap_bbox, tensor2img, \
      sampled_margin_rank_loss, computeMatchmap, vis_matchmap, infoNCE_loss, measure_pVA
from utils.tf_equivariance_loss import TfEquivarianceLoss

from datetime import datetime, timedelta


from utils.utils import save_checkpoint, AverageMeter,  \
     Logger, neq_load_customized, ProgressMeter

import psutil

def get_total_memory_gb():
    with open('/proc/meminfo', 'r') as f:
        meminfo = f.read()
    for line in meminfo.split('\n'):
        if "MemTotal" in line:
            mem_kb = int(line.split()[1])
            return mem_kb / 1024 / 1024  # Convert to GB
    return None


def get_total_memory_gb_2():
    """
    Returns the total RAM memory in GB using psutil.
    """
    mem = psutil.virtual_memory()
    return mem.total / (1024 ** 3)  # Convert bytes to GB

def get_physical_cores():
    cores = set()
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
    for block in cpuinfo.strip().split('\n\n'):
        core_id = None
        for line in block.split('\n'):
            if line.startswith("core id"):
                core_id = int(line.split(":")[1])
            elif line.startswith("physical id"):
                physical_id = int(line.split(":")[1])
        if core_id is not None:
            cores.add((physical_id, core_id))
    return len(cores)



def is_running_on_cluster():
    # Check for common cluster environment variables
    cluster_env_vars = ['SLURM_JOB_ID', 'PBS_JOBID', 'SGE_TASK_ID']
    if any(var in os.environ for var in cluster_env_vars):
        return True
    return False

def set_path(args):
    # Create the experiment folder
    if is_running_on_cluster(): 
        exp_path = os.path.expandvars('$SCRATCH/TEST_{args.exp_name}'.format(args=args))
    else:
        exp_path = 'garbage/TEST_{args.exp_name}'.format(args=args) #if not using the cluster

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    else:
        print('The experiment folder already exists')
        
    img_path = os.path.join(exp_path, 'img') 
    model_path = os.path.join(exp_path, 'model') 

    if not os.path.exists(img_path): 
        os.makedirs(img_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)
    return img_path, model_path, exp_path

def check_if_available(data_json):
        """
        Check if 7 seven days have passed since the time creation
        """
        # Parse the time_creation string into a datetime object
        creation_time = datetime.strptime(data_json["parameters"]["time_creation"], "%Y-%m-%d %H:%M")

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

def batch_unpacker(batch, args):
    if args.punish_silence:
        if args.get_nFrames:
            image, spec, audiofile, silence_vector, nFrames = batch
        else:
            image, spec, audiofile, silence_vector = batch
            nFrames = None
    else:
        image, spec, audiofile = batch
        silence_vector = None
        nFrames = None
    return image, spec, audiofile, silence_vector, nFrames


def validate(val_loader, model, criterion, device, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    pVA_aud_meter = AverageMeter()
    pVA_pad_meter = AverageMeter()
    pVA_meter = AverageMeter()
    Nt_pad = AverageMeter()
    Nt_aud = AverageMeter()
    
    img_embs_all = []
    aud_embs_all = []

    

    tic = time.time()
    save_dir = os.path.join(args.img_path, "test_imgs", str(epoch)) 

    model.eval()
    
    video_gen = False
    with torch.no_grad():
        end = time.time()
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            image, spec, audio_path, silence_vectors, nFrames = batch_unpacker(batch,args)

            spec = Variable(spec).to(device, non_blocking=True)
            image = Variable(image).to(device, non_blocking=True)
            silence_vectors = Variable(silence_vectors).to(device,non_blocking=True)  if args.punish_silence else None
            B = image.size(0)
            # vis_loader(image, spec,idx)

            imgs_out, auds_out = model(image.float(), spec.float(), args, mode='val')

            imgs_out = imgs_out.to('cpu').detach()
            auds_out = auds_out.to('cpu').detach()
 

            loss_cl = infoNCE_loss(imgs_out,auds_out, args)
            
            if args.punish_silence:
                pVA_aud, pVA_pad, pVA, Nt_aud, Nt_pad = measure_pVA(silence_vectors, imgs_out, auds_out, nFrames)
                pVA_aud_meter.update(pVA_aud.item(), B)
                pVA_pad_meter.update(pVA_pad.item(), B)
                pVA_meter.update(pVA.item(), B)
                Nt_aud.update(Nt_aud.item(), B)
                Nt_pad.update(Nt_pad.item(), B)
            
                

                if args.use_wandb:
                    wandb.log({"pVA_aud_step": pVA_aud.item(), "step": wandb.run.step})
                    wandb.log({"pVA_pad_step": pVA_pad.item(), "step": wandb.run.step})
                    wandb.log({"pVA_step": pVA.item(), "step": wandb.run.step})
                    wandb.log({"Nt_aud_step": Nt_aud.item(), "step": wandb.run.step})
                    wandb.log({"Nt_pad_step": Nt_pad.item(), "step": wandb.run.step})

                #TODO: measure mIoU
            
            if args.cross_modal_freq != -1 and (epoch % args.cross_modal_freq) == 0:
                img_embs_all.append(imgs_out)
                aud_embs_all.append(auds_out)

            losses.update(loss_cl.item(), B)
           
            batch_time.update(time.time() - end)
            end = time.time()
            

            if args.video and (epoch % args.val_video_freq) == 0:
                if not video_gen:
                    if B*idx <= args.val_video_idx < idx*B + B:
                        t_make_video =time.time()
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
                        print(f" - Time elapsed for creating the video: {time.time() - t_make_video:.2f}")
                        video_gen = True


                        if args.use_wandb:
                            t_upload_video = time.time()
                            wandb.log({"val_video": wandb.Video(video_dir, caption=f"Epoch {epoch}"), "epoch": epoch})
                            print(f" - Time elapsed for uploading the video to wandb: {time.time() - t_upload_video:.2f}")
    print('Epoch: [{0}]\t Eval '
          'Loss: {loss.avg:.4f}  \t T-epoch: {t:.2f} \t'
          .format(epoch, loss=losses, t=time.time()-tic))


    if args.cross_modal_freq != -1 and (epoch % args.cross_modal_freq) == 0:
        imgs_out_all = torch.cat(img_embs_all)
        auds_out_all = torch.cat(aud_embs_all)

        print("\tAbout to calculate the sims")
        sims = similarity_matrix_bxb(imgs_out_all,auds_out_all)
        
        recalls    = topk_accuracies(sims, [1,5,10])
        A_r10 = recalls["A_r10"]
        A_r5  = recalls["A_r5"]
        A_r1  = recalls["A_r1"]
        I_r10   = recalls["I_r10"]
        I_r5    = recalls["I_r5"]
        I_r1    = recalls["I_r1"]
        
        if args.use_wandb:
            wandb.log({
                "val_loss": losses.avg,
                "epoch": epoch,
                "A->V R@10": A_r10,
                "A->V R@5": A_r5,
                "A->V R@1": A_r1,
                "V->A R@10": I_r10,
                "V->A R@5": I_r5,
                "V->A R@1": I_r1
            })

        N_examples= len(val_loader)*B
    else:
        if args.use_wandb:
            wandb.log({
                "val_loss": losses.avg,
                "epoch": epoch
            })

    
    if args.cross_modal_freq != -1 and (epoch % args.cross_modal_freq) == 0:
        print(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
            .format(A_r10=A_r10, I_r10=I_r10, N=N_examples), flush=True)
        print(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
            .format(A_r5=A_r5, I_r5=I_r5, N=N_examples), flush=True)
        print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
            .format(A_r1=A_r1, I_r1=I_r1, N=N_examples), flush=True)
        
    if args.punish_silence:
        print('-  * pVA_aud {pVA_aud:.3f} pVA_pad {pVA_pad:.3f} pVA {pVA:.3f}'
            .format(pVA_aud=pVA_aud_meter.avg, pVA_pad=pVA_pad_meter.avg, pVA=pVA_meter.avg), flush=True)
        print('-  * Nt_aud {Nt_aud:.3f} Nt_pad {Nt_pad:.3f} over {N:d} validation pairs'
            .format(Nt_aud=Nt_aud.avg, Nt_pad=Nt_pad.avg, N=N_examples), flush=True)
        
        
    
    print(f"Time elapsed in total{time.time()-tic}")

    return losses.avg, 0, 0


def main(args):
    print(f"-----------------Testing experiment {args.exp_name}-----------------")
    
    print("Is cuda available",torch.cuda.is_available()) 

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))

    if args.use_cuda:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please run with flag --use_cuda False")

    print('Using GPU:', args.gpus)
    print('Number of of cores allocated:',len(os.sched_getaffinity(0)))
    print(f"{os.sched_getaffinity(0)} CPU cores that job {args.job_id} is allowed to run on")  # Will show the CPU cores your job is allowed to run on
    print(f"Total RAM memory: {len(os.sched_getaffinity(0))*15.9} GB")
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
    
    if torch.cuda.is_available():
        # device = torch.device('cuda:1') if len(args.gpus) > 1 else torch.device('cuda:0')
        device = torch.device('cuda:0') if len(args.gpus) > 1 else torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Check if the JSON file exists
    if not os.path.exists(args.links_path):
        raise FileNotFoundError(f"The JSON file at {args.links_path} does not exist.")
    
    #Open json ----------------------------------- EVERYTHING IS IN THE JSON
    with open(args.links_path, 'r') as f:
        epochs_data = json.load(f)

    print("\tEpochs to test: ", len(epochs_data)-1)
    keys = list(epochs_data.keys())
    print(f"\tFirst epoch {keys[1]}, Last epoch {keys[-1]}")

    #Check if it's still available to use (7 days after its creation)
    if epochs_data["parameters"]== "Jose Antonio Santos":
        args.get_nFrames = True
    elif not check_if_available(epochs_data):
        raise RuntimeError("The model is no longer available for testing. It has exceeded the 7-day availability period.")

    #Dataset
    args.img_path, args.model_path, args.exp_path = set_path(args)
    
    val_dataset = PlacesAudio(args.placesAudio + 'val.json', args,mode='val')
    
    print("Creating Data loaders with %d workers" % args.n_threads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.n_threads, drop_last=True, pin_memory=True)

    if args.MISA_2_LVS_epoch != 0:
        print(f"MISA_2_LVS_epoch at {args.MISA_2_LVS_epoch}")
        error = False
        string_error = "MISA_2_LVS_epoch is set to {args.MISA_2_LVS_epoch}" 
        if args.simtype != "MISA":
            string_error += f" but simtype is set to {args.simtype}"
            error = True
        if args.LVS:
            string_error += " but LVS is set to True and it should be False"
            error = True
        if error:
            raise ValueError(string_error)
    
        
    #Test every model in the json
    for epoch in keys[1:]:
        if args.SISA_2_MISA_epoch != 0 and args.SISA_2_MISA_epoch <= int(epoch):
            print(f" - Changing from {args.simtype} to MISA at epoch {epoch} -")
            args.simtype = 'MISA'
            args.SISA_2_MISA_epoch = 0

        if args.MISA_2_LVS_epoch != 0 and args.MISA_2_LVS_epoch <= int(epoch):
            print(f" - Changing from {args.simtype} to LVS at epoch {epoch} -")
            args.LVS = True
            args.MISA_2_LVS_epoch = 0

        model, device = load_model(epochs_data[epoch]['weights_link'],args)
        validate(val_loader,model,None,device,int(epoch),args)

    print("Finished")
    sys.exit(0)

if __name__ == "__main__":
    args=get_arguments()
    main(args)







