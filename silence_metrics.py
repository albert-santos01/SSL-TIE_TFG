import re
from utils.utils import AverageMeter
import time
import torch
from torch.autograd import Variable
from tqdm import tqdm
from utils.util import *
import os
from datasets import PlacesAudio
from torch.utils.data import  DataLoader
import pandas as pd


def extract_epoch_number(file_path):
    """
    Extracts the epoch number from a model weight file path.

    Args:
        file_path (str): The file path of the model weight file.

    Returns:
        int: The epoch number if found, otherwise None.
    """
    match = re.search(r'epoch(\d+)', file_path)
    if match:
        return int(match.group(1))
    return None

def get_silence_results(val_loader, model, device,epoch,args):
    pVA_aud_meter = AverageMeter()
    pVA_pad_meter = AverageMeter()
    pVA_meter = AverageMeter()
    sil_loss_meter = AverageMeter()


    model.eval()

    with torch.no_grad():
            
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                image, spec, _, silence_vectors, nFrames = batch_unpacker(batch,args)

                spec = Variable(spec).to(device, non_blocking=True)
                image = Variable(image).to(device, non_blocking=True)
                silence_vectors = Variable(silence_vectors).to(device,non_blocking=True)  if args.punish_silence else None
                B = image.size(0)
            
                imgs_out, auds_out = model(image.float(), spec.float(), args, mode='val')

                imgs_out = imgs_out.to('cpu').detach()
                auds_out = auds_out.to('cpu').detach()
                T = auds_out.size(2)

                loss_sil, normal_mean = negAudio_loss(imgs_out,auds_out,silence_vectors)

                pVA_aud, pVA_pad, pVA, _, _ = measure_pVA(silence_vectors, imgs_out, auds_out, nFrames)
                
                pVA_aud_meter.update(pVA_aud.item())
                pVA_pad_meter.update(pVA_pad.item())
                pVA_meter.update(pVA.item())

                sil_loss_meter.update(loss_sil.item())
                print(f"Step: {idx}, pVA_aud: {pVA_aud.item()}, pVA_pad: {pVA_pad.item()}, pVA: {pVA.item()}, sil_loss: {loss_sil.item()}")

    if args.use_wandb:
        wandb.log({
            "pVA_aud": pVA_aud_meter.avg,
            "pVA_pad": pVA_pad_meter.avg,
            "pVA": pVA_meter.avg,
            "sil_loss" : sil_loss_meter.avg, 
            "epoch": epoch
        })

    return pVA_aud_meter.avg, pVA_pad_meter.avg, pVA_meter.avg, sil_loss_meter.avg

def main(models_path):
    
    results = {}

    sys.argv = ['script_name', '--order_3_tensor', '--spec_DAVENet', '--padval_spec', '-80',
                '--placesAudio', '$DATA/PlacesAudio_400k_distro/metadata/',
                '--n_threads', '4',
                '--batch_size', '200',
                '--punish_silence',
                '--get_nFrames'
                ]


    args = get_arguments()

    # get the val_loader

    val_dataset = PlacesAudio(args.placesAudio + 'val.json', args,mode='val')

    print("Creating Data loaders with %d workers" % args.n_threads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.n_threads, drop_last=True, pin_memory=True)

    # Walk through the directory and get all files
    
    for root, _, files in os.walk(models_path):
        for model_path in files:
            epoch = extract_epoch_number(model_path)
            
            #Load model
            model, device = load_model(os.path.join(root,model_path),args)
            
            pVA_aud, pVA_pad, pVA, sil_loss = get_silence_results(val_loader, model, device, epoch, args)

            #Store in results
            results[model_path] = {
                "pVA_aud": pVA_aud,
                "pVA_pad": pVA_pad,
                "pVA": pVA,
                "sil_loss": sil_loss,
                "epoch": epoch
            }
            print(f"Model: {model_path}, pVA_aud: {pVA_aud}, pVA_pad: {pVA_pad}, pVA: {pVA}, sil_loss: {sil_loss}")
            

    # Save results to a csv file

    # Define the file path
    file_path = 'silence_results.csv'
    # Create a DataFrame from the results dictionary
    df = pd.DataFrame.from_dict(results, orient='index')

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=True)
    

if __name__ == "__main__":
    models_path = "/home/asantos/models/to_test"
    main(models_path)
