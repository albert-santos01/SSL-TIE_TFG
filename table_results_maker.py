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
import DAVEnet_models


def extract_model_name(file_path):
    """
    Extracts the model name from a file path.
    Example: /home/asantos/models/to_test/new_split/SSL_TIE_PlacesAudio_split-lr1e-4-B128-fMISA-SdT128-epoch49.pth.tar
    Will return: VGS-TIE-fMISA
    Args:
        file_path (str): The file path of the model weight file.
    
    Returns:
        str: The model name extracted from the file path.
    """
    model_file = file_path.split('/')[-1]
    modes = model_file.split('-')
    if len(modes) == 7:
        model_name = f"VGS-TIE-{modes[4]}"
    elif len(modes) == 8:
        model_name = f"VGS-TIE-{modes[4]}-{modes[6]}"
    return model_name
    
    

def  get_results_vgs_tie(val_loader, model, device, name, args,results):
    pVA_aud_meter = AverageMeter()
    pVA_pad_meter = AverageMeter()
    pVA_meter = AverageMeter()
    
    img_embs_all = []
    aud_embs_all = []

    model.eval()
    
    video_gen = False
    with torch.no_grad():

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

            #For crossmodal retrieval
            img_embs_all.append(imgs_out)
            aud_embs_all.append(auds_out)
            
            pVA_aud, pVA_pad, pVA, _, _ = measure_pVA(silence_vectors, imgs_out, auds_out, nFrames)
            pVA_aud_meter.update(pVA_aud.item())
            pVA_pad_meter.update(pVA_pad.item())
            pVA_meter.update(pVA.item())


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

        print("\n\n\tResults for %s" % name)
        print("Silence: ")
        print("pVA: %.4f, pVA_pad: %.4f, pVA_aud: %.4f" % (pVA_meter.avg, pVA_pad_meter.avg, pVA_aud_meter.avg))
        print("Crossmodal retrieval: ")
        print("A_r10: %.4f, A_r5: %.4f, A_r1: %.4f" % (A_r10, A_r5, A_r1))
        print("I_r10: %.4f, I_r5: %.4f, I_r1: %.4f" % (I_r10, I_r5, I_r1))

    results[name] = {
        "pVA": pVA_meter.avg,
        "pVA_pad": pVA_pad_meter.avg,
        "pVA_aud": pVA_aud_meter.avg,
        "A_r10": A_r10,
        "A_r5": A_r5,
        "A_r1": A_r1,
        "I_r10": I_r10,
        "I_r5": I_r5,
        "I_r1": I_r1
    }

    return results
    


def get_silence_results_davenet(val_loader, image_model, audio_model, device, args, results):
    pVA_aud_meter = AverageMeter()
    pVA_pad_meter = AverageMeter()
    pVA_meter = AverageMeter()

    image_model.eval()
    audio_model.eval()

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                image, spec, _, silence_vectors, nFrames = batch_unpacker(batch,args)

                spec = Variable(spec).to(device, non_blocking=True)
                image = Variable(image).to(device, non_blocking=True)
                silence_vectors = Variable(silence_vectors).to(device,non_blocking=True)  if args.punish_silence else None
                B = image.size(0)
            
                imgs_out = image_model(image)
                imgs_out = imgs_out.to('cpu').detach() 

                auds_out = audio_model(spec)
                auds_out = auds_out.to('cpu').detach()

                T = auds_out.size(2)

                pVA_aud, pVA_pad, pVA, _, _ = measure_pVA(silence_vectors, imgs_out, auds_out, nFrames)
                pVA_aud_meter.update(pVA_aud.item())
                pVA_pad_meter.update(pVA_pad.item())
                pVA_meter.update(pVA.item())


    A_r10   = 0.286
    A_r5    = 0.199
    A_r1    = 0.064
    I_r10   = 0.263
    I_r5    = 0.166
    I_r1    = 0.052

    print("\n\n\tResults for DAVEnet")
    print("Silence: ")
    print("pVA: %.4f, pVA_pad: %.4f, pVA_aud: %.4f" % (pVA_meter.avg, pVA_pad_meter.avg, pVA_aud_meter.avg))
    print("Crossmodal retrieval: ")
    print("A_r10: %.4f, A_r5: %.4f, A_r1: %.4f" % (A_r10, A_r5, A_r1))
    print("I_r10: %.4f, I_r5: %.4f, I_r1: %.4f" % (I_r10, I_r5, I_r1))
    results["DAVEnet"] = {
        "pVA": pVA_meter.avg,
        "pVA_pad": pVA_pad_meter.avg,
        "pVA_aud": pVA_aud_meter.avg,
        "A_r10": A_r10,
        "A_r5": A_r5,
        "A_r1": A_r1,
        "I_r10": I_r10,
        "I_r5": I_r5,
        "I_r1": I_r1
    }

    return results


                



def load_DAVEnet(paths,device):
    """
    Load the DAVEnet model from the specified paths.
    
    Args:
        paths (list):  List of paths to the DAVEnet model files.
                        [image_path, audio_path]
        device (torch.device): The device to load the model onto (CPU or GPU).
    Returns:
        image_model (torch.nn.Module): The image model of DAVEnet.
        audio_model (torch.nn.Module): The audio model of DAVEnet.
    """

    image_model = nn.DataParallel(DAVEnet_models.VGG16().to(device))
    audio_model = nn.DataParallel(DAVEnet_models.Davenet().to(device))

    image_model.load_state_dict(torch.load(paths[0], map_location=device))
    audio_model.load_state_dict(torch.load(paths[1], map_location=device))

    return image_model, audio_model


def main(models_path,davenet_paths):

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    #TODO Get Crossmodal retrieval results and Silence results
       # First DAVEnet
       # Then all the other models

    # DAVEnet model 
    sys.argv = ['script_name', '--order_3_tensor', '--spec_DAVENet', '--padval_spec', '0',
                '--placesAudio', '$DATA/PlacesAudio_400k_distro/metadata/',
                '--n_threads', '4',
                '--batch_size', '200',
                '--punish_silence',
                '--get_nFrames'
                ]


    args_dave = get_arguments()

    val_dataset_dave = PlacesAudio(args_dave.placesAudio + 'val.json', args_dave, mode='val')

    print("Creating Data loaders with %d workers" % args_dave.n_threads)
    val_loader_dave = DataLoader(val_dataset_dave, batch_size=args_dave.batch_size, shuffle=False,\
        num_workers=args_dave.n_threads, drop_last=True, pin_memory=True)
    
    # Load DAVEnet model
    image_model, audio_model = load_DAVEnet(davenet_paths, device)

    # Get Silence results for DAVEnet
    results = get_silence_results_davenet(val_loader_dave, image_model, audio_model, device, args_dave, results)



    

    # VGS-TIE models
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
            name = extract_model_name(model_path)

            #Load model
            model, device = load_model(os.path.join(root,model_path),args)

            results = get_results_vgs_tie(val_loader, model, device, name, args,results)

    return results

    
if __name__ == "__main__":
    models_path = '/home/asantos/models/to_test/new_split/'
    davenet_paths = [
        '/home/asantos/code/checkpoints/image_model.136.pth',
        '/home/asantos/code/checkpoints/audio_model.136.pth'
    ]
    results_dir = '/home/asantos/results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Run the main function to get results
    print("Starting the main function to get results...")
    results = main(models_path, davenet_paths)
    

    print("Results obtained. Now saving to CSV...")
    # Convert results to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')

    # Define the file path - do not overwrite existing files Create a new file if it exists
    existing_files = [f for f in os.listdir(results_dir) if f.startswith('silence_results') and f.endswith('.csv')]
    if existing_files:
        latest_file = max(existing_files, key=lambda x: int(re.search(r'(\d+)', x).group(1)))
        latest_index = int(re.search(r'(\d+)', latest_file).group(1))
        new_index = latest_index + 1
    else:
        new_index = 0
    file_name = f'silence_results_{new_index}.csv'

    file_path = os.path.join(results_dir, file_name)
    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=True)
    print(f"Results saved to {file_path}")
    print("Done!")
    




