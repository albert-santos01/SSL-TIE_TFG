# Study of the code
Here we study the code of the model SSL-TIE. We start with the `main.py` file in order to run the model and see if it works. The procedure will be to run the model with the provided weights and data; maybe we test it and provide the results to verify that the model is working as expected.

## MAIN main.py
directly to def main() function, [`main.py`, line 446](./main.py#L446)

```python
def main(args):
    if args.gpus is None:
        args.gpus = str(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpus)
        args.gpus = list(range(torch.cuda.device_count()))

    if args.debug:
        args.n_threads=0

    args.host_name = os.uname()[1]
    device = torch.device('cuda:1') if len(args.gpus) > 1 else torch.device('cuda:0')

    best_acc = 0
    best_miou = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.img_path, args.model_path, args.exp_path = set_path(args)

    model = AVENet(args)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=args.gpus, output_device=device)  
    model_without_dp = model.module

    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[300,700,900], gamma=0.1)
    args.iteration = 1
```

AVENet is the model being used which comes as imported from `model.py` [`main.py`, line 29](./main.py#L29)

It is important to note that the code assumes an evironment variable `CUDA_VISIBLE_DEVICES` to be set. If it is not set, it will raise an error to set it to `args.gpus`, changed at the commit of [Edit to handle the gpu compability](https://github.com/jinxiang-liu/SSL-TIE/commit/9d753366b3dc5e3b16cb3169fbc185ae3a1416ac) alongside with some modifications to ensure the initialization of Cuda context.

Next part of the main.py for testing:
```python
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

            logger_path = os.path.abspath(os.path.join('../img/logs/test', os.path.dirname(args.test))) # modified
            # logger_path = os.path.join(args.img_path, 'logs', 'test')
            if not os.path.exists(logger_path):
                os.makedirs(logger_path)


            args.test_logger = Logger(path=logger_path)
            args.test_logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
```

Here basically the code is loading a checkpoint file and testing the model with the provided weights. The logger is used to log the results of the test. The loader assumes the weights, in this case, alledgely a tar file `vggsound144k.pth.tar`. However, the logger has been modified to work with windows, errors like invalid path were not expected. Commit [Change logger to work with windows](https://github.com/albert-santos01/SSL-TIE_TFG/commit/749b6f0d91ef258442affeaf3cc5b8224675ff89) was made to fix this issue. 
- Also it is strange that they want use the directory of the checkpoint to name the logger

Next, the code will load the data and start the test
```python
        if args.dataset_mode == 'VGGSound':
            test_dataset = GetAudioVideoDataset(args, mode='test' if args.test_set == 'VGGSS' else 'val')
        elif args.dataset_mode == 'Flickr':
            test_dataset = GetAudioVideoDataset(args, mode='test')

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,\
            num_workers=args.n_threads, pin_memory=True)
        
        test(test_loader, model, criterion, device, epoch, args )
```	
[`main.py, Line 512`](./main.py#L512)

Where the definition of GetAudioVideoDataset is in the file `dataloader.py` at `datasets` folder. The following is its definition, with the relevant part for the test set, the rest is hidden for brevity.

### GetAudioVideoDataset (dataloader.py)
```python
class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
 
        data = []
        self.args = args

        if args.dataset_mode == 'VGGSound':
            args.trainset_path = args.trainset_path
        elif args.dataset_mode == 'Flickr':
            args.trainset_path = args.Flickr_trainset_path

        # Debug with a small dataset
        if args.debug: # what they do is to track specific images I think
           ...

        else:
            if args.dataset_mode == 'VGGSound':
                if mode=='train':
                  ...
                elif mode=='test':
                    with open('metadata/test_vggss_4911.txt','r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader[:]:
                            data.append(item.split('.')[0])
                        self.audio_path = args.vggss_test_path + '/audio/'
                        self.video_path = args.vggss_test_path + '/frame/'
                elif ...

        self.imgSize = args.image_size 

        self.AmplitudeToDB = audio_T.AmplitudeToDB()

        self.mode = mode
        self.transforms = transforms
        # initialize video transform
        self._init_atransform()
        self._init_transform()
        #  Retrieve list of audio and video files
        self.video_files = []

        for item in data[:]:
            self.video_files.append(item )

        print("{0} dataset size: {1}".format(self.mode.upper() , len(self.video_files)))
        
        self.count = 0 
    ...

    def __len__(self):
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        file = self.video_files[idx]

        if self.args.dataset_mode == 'VGGSound':
            if self.mode == 'train':
                ...
            elif self.mode in ['test', 'val'] :
                frame = self.img_transform(self._load_frame( os.path.join(self.video_path , file + '.jpg')  ))
                frame_ori = np.array(self._load_frame(os.path.join(self.video_path, file + '.jpg')))
                samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.wav'))
        
        ### For Flickr_SoundNet training: 
        elif self.args.dataset_mode == 'Flickr':
            ...

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
        
        if (self.args.aud_aug=='SpecAug') and (self.mode=='train') and (random.random() < 0.8):
            maskings = nn.Sequential(
                audio_T.TimeMasking(time_mask_param=180),
                audio_T.FrequencyMasking(freq_mask_param=35)
                )
            spectrogram = maskings(spectrogram)

        spectrogram = self.AmplitudeToDB(spectrogram)


        return frame, spectrogram, 'samples', file, torch.tensor(frame_ori)
```
[`dataloader.py, Line 44`](./datasets/dataloader.py#L44)

Presumably here is where the tranformations begin and the data is loaded. Things to note:
- The text file `metadata/test_vggss_4911.txt` contains at each line the name of the file like `nqd6uO6hDSo_000030.jpg` and then this is split to get the name of the file without the extension saved at the attribute `video_files`.

Later the main.py creates the DataLoader object with the test_dataset that obtained from the GetAudioVideoDataset class; and this new object is imported from the torch.utils.data library. An idea of this process could be obtained from the following webpage: [Pytorch DataLoader, Creating a Custom Dataset for your files](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)

What we can understand from this is that GetAudioVideoDataset is a class that inherits from the Dataset class from the torch.utils.data library. This one its attibrutes and methods will create the dataset as desired. From this one could consider that:
- the data has to be stored at the `args.vggss_test_path` directory ('dir_of_vggsound_testset')
- the other two important methods are `__len__` and `__getitem__` which are used to get the length of the dataset and to get the data at a specific index respectively.
- From the `__getitem__` method one can see that the method returns the frame, the spectrogram, the samples, the file name and the original frame.
- Everything should be stored at the `args.vggss_test_path` directory because video_path and audio_path are defined as that.
- With the audio, (the samples operation) they are repeating the audio to have a length of at least 10 seconds, then they take the first 10 seconds of the audio.
- The frame goes through the img_transform method which is defined in the same file, and the audio goes through the MelSpectrogram transformation which is defined in the `audio_T` module.


Thins to do Right NOW:
- [ ] Try to run the model again
- [ ] Upload the test set
- [ ] Upload the weights
- [ ] Run the model with the test set and weights
- [ ] Check the logger and see if it works

Aparently everything is working, but from now on I haven't downloaded nothing about the data.

### VGGSound dataset
The VGGSound dataset is a large-scale dataset for audio-visual learning, containing 200,000 YouTube videos with more than 310 classes. To download the dataset you need the csv file `vggsound.csv` which is available at the [VGGSound website](https://www.robots.ox.ac.uk/~vgg/data/vggsound/). Each line of the csv file contains:
```
# YouTube ID, start seconds, label, train/test split. 
```
So with this information one can download the videos from YouTube and then extract the audio and frames from the videos. To do that, it is recommended to use [audiosetdl](https://github.com/speedyseal/audiosetdl), which is a repository from the University of Oxford that provides modules and scripts to download Google's AudioSet dataset.

It seems that it will take us a long time and we need to create scripts to download the data

### Flickr-SoundNet dataset
Due to the time spent and the complexity of the VGGSound dataset, we decided to use the Flickr-SoundNet dataset instead just for doing the test of the model.

The Flickr-SoundNet is provided by Senocak in the repository [learning_to_localize_sound_source](https://github.com/ardasnck/learning_to_localize_sound_source). The dataset in order to be used for testing it should be stored in a folder called `dir_of_SoundNet_Test_Data` as demaned in the [test_Flickr10.sh](./scripts/test_Flickr10k.sh). 
The instances required for testing are written in the ``test_flickr_250.txt``, regardless the amount inquired by the script. 
```python

class GetAudioVideoDataset(Dataset):
    def __init__(self, args, mode='train', transforms=None):
        ...
            elif args.dataset_mode == 'Flickr':
                if mode == 'train': ...
                elif mode == 'test':
                    with open('metadata/test_flickr_250.txt','r') as f:
                        txt_reader = f.readlines()
                        for item in txt_reader:
                            data.append(item.split('.')[0])
                        self.audio_path = args.soundnet_test_path + '/mp3/'
                        self.video_path = args.soundnet_test_path + '/frame/'
    ...

    def __getitem__(self, idx):
        file = self.video_files[idx]
        ### For Flickr_SoundNet training: 
        elif self.args.dataset_mode == 'Flickr':
            if self.mode == 'train': ...
            elif self.mode in ['test', 'val'] :
                frame = self.img_transform(self._load_frame( os.path.join(self.video_path , file + '.jpg')  ))
                frame_ori = np.array(self._load_frame(os.path.join(self.video_path, file + '.jpg')))
                samples, samplerate = torchaudio.load(os.path.join(self.audio_path, file + '.mp3'))
```
[`dataloader.py, Line 44`](./datasets/dataloader.py#L44)

Therefore `jpg` and `mp3` files are expected to be found in the `dir_of_SoundNet_Test_Data` directory. The `test_flickr_250.txt` file contains the names of the files without the extension.
