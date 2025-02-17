### 08/11/2025: Setup and Issues with VGGSound

#### **Summary of Work**
- Installed `ffmpeg` and Python on WSL.
- Tried setting up the code for installing VGGSound but faced significant issues due to compatibility and outdated dependencies.

#### **Problems Encountered**
- The repository for VGGSound is MacOS-focused, causing challenges in WSL.
- Conflict between `youtube-dl` and ``yt-dlp``:
    - **Issue**: Repository uses ``pafy`` with ``youtube-dl`` as the backend, which is outdated and doesn't work properly.

#### **Attempts Made**
- Installed ``youtube-dl`` and ``pafy`` on WSL → Unable to find videos.
- Installed ``miniconda`` as suggested with the corresponding Python requirements → Output was ``None``.
- Installed ``yt-dlp`` and manually tested it → Successfully found and downloaded videos but to be implemented for the whole dataset requires major code changes.

#### **Next Steps**
- Option 1: Use Flickr-SoundNet instead of VGGSound.
- Option 2: Ask Xavier for guidance on VGGSound.
- Option 3: Continue working on VGGSound in WSL:
    - Replace pafy code with alternatives like import yt_dlp as pafy.
    - Rewrite the entire code to use yt-dlp.

#### **Time Spent**
The whole day

### 09/11/2025: 
I just decided that I will use Flickr-SoundNet instead of VGGSound.
1. Understand how the code uses this dataset. Now it is written at the study of the code
2. I proved that the dataset is unique and has all the videos that I need, the ones that ``test_flickr_250.txt`` demands
3. Now everything is stored correctly for the Flickr-SoundNet dataset. Now it is time to test the model!

I managed to test the model with Flickr10k, proof:
```shell
TEST dataset size: 249
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 249/249 [01:04<00:00,  3.89it/s]
Test:    Epoch: [33]    Loss: 0.0375 Acc@1: 0.0000 Acc@5: 0.0000 MeancIoU: 0.7791 AUC: 0.5927
```
Important things to note:
- I had to change the audio format from ``.wav`` to ``.mp3`` because the repo wants mp3 but torchaudio with soundfile as a backend doesn't support mp3. Since the change of extension should not affect mininmally the audio, I decided to do it. Remember that we use soundfile as a backend because since I use Windows, I can't use ``sox_audio``  for ``torchaudio==0.7.2``.
- I had to change the number of workers from 16 to 8 because it was causing a memory error. The repo origanally could handle 16 workers, but my computer has only 8 cores.
- I had to also upload the xml annotations.
Then finally I could test the model with the Flickr-SoundNet dataset.

Tomorrow:
Understand the results

### 12/11/2025:

I read again the SSL-TIE paper. What I get from reading it this time,  is that I tested correctly the Flickr10k model and
I assume that we don't have the exact same results because of the change from `mp3` to `wav`. However, the paper states that audio are cutted
till 20 segs, which it is noticeable with the wav audios but no with mp3. The reading also inspired me to write the Related Work.

It is important to note that by reading again the paper of Xavier, we could apply the same idea for extending the datasets and Then
test our new version of SSL-TIE against their metrics of negative audio. Moreover, from checking its github I noticed that models are easily
deployed. Giving me the idea that when it means that the backbone is ResNet-18 you simply puts its weights and there you have it, I still
have to check if for example at the SSL-TIE the Siamese architechure is build in this repository.

Conversely, what makes a model different from the other is its training. All of them adopt different ways to apply data augmentation principles
and its losses functions, (The way that the model learns). Hence, to test is basically forward, get its output and analyse it against a benchmark.


### 13/01/2025:

Plan of the day:
1. Read the Dense AV
2. See if I can download PlacesAudio
3. Catch up from what gloria and I talked about doing at the end
4. read Clip-denoiser
5. have a plan for tomorrows meeting
6. write letter to the bsc and email to inetum.

By reading again DenseAV's, it is easy to recognize that this model is huge and relies on a lot of parameters since it has multihead attention. Still I have to understand what is the cls token, but both backbones are transformers. One interesting point is that penalizes when multiple attention head are activated(3.4 disentaglement Regularizer).

**2016 - Harwath - (PlacesAudio) Unsupervised Learning of Spoken Languag with Visual Context**
After reading it, I was lead to read the paper of PlacesAudio, which in it describes a model  which is capable of associating speech audio captions with images. Amazingly, it does not require a such big architechure compared to the DenseAV, so it gives hope for the development of the thesis.

Regarding the dataset, this one is obtained from a subset of Places 205, which is 2.5M images categorized into 205 classes. Then with Spoke JavaScript framework and with Google SpeechRecognition they manage to obtain 120.000 captions. These are published and splitted into 144.000 training set, a 2,400 development set(maybe val set), and 2.400 for test set. The average speech form caption is 9.5 secs, containing an average of 21.9 words.

**2018 - Harwarth - Jointly Discovering Visual Objects and Spoken Words**

Now in this paper, PlacesAudio had been extended to 402.835 image/caption pairs. Here they also use ADE20k image/speech caption whose underlying scene category is indeed in the Places 205 label set.


### 14/01/2025
Cosas que quiero hacer hoy:
- Hacer el log de reunion de hoy
- Enviar Carta BSC y correo a Inetum
- Hacer Roadmap de lo siguiente del TFG
- Mirar Vuelos y dedicar tiempo a pensar que puedo hacer
- Hacer algo mas del TFG

Today, me and Gloria had a meeting to talk about the progress of the project. We declared that the main aim  of the thesis  is to try and adapt SSL-TIE to get the audio temporal analysis as DenseAV or DAVENet.
Basically, the same thing of DAVENet but with SSL-TIE, ideal to get Natural Sounds as well, with ADE20k

Next, in order to improve it we can use DINO or CLIP-DINOiser because it does a finer localisation and segmantic localization, And do the same thing as DenseAV or CLIP-DINOiser. Add this features to our SSL-TIE.

Finally, do a huge aplication that by speaking or making sounds it will localize the considered subject. This is the final part of the thesis which indeed it will take time as well. This can be done with flask xd. I need to work A LOT bruv.

Therefore, here the roadmap:
#### ROADMAP, option 1:
1. Get access to the cluster and understand how it works
2. Create whole directory in the cluster
3. Create sbatch file in order to run (You have the support document that Xavier gave you)
4. I would run first a test
5. Upload somehow the datasets (PlacesAudio) and ADE20k
7. Analize how to get the temporal audio
6. Do a single epoch
7. Estimate time to do the whole dataset
If it's not to much
8. Train with PlacesAudio
9. Meanwhile create a script to see the work and analyse it
10. Then train with both (ADE20K and PlacesAudio)
11. See results with both
Else:
8. train with both
9. see results with both with the script done with both

#### ROADMAP option 2:
Basically prepare a subset and run them in my home computer

More things to mention about the meeting (because i remember on the while):
- My results of testing SSL-TIE are probably different because I tested it with a Windows and I've changed things for the code to run in windows, and Xavier suggests that the results are different probably because of this. It is recommendable to change the whole environment to ubuntu xdddd.

### 03/02/2025
I managed to get till the 5th milestone. I encountered many problems that I briefly mention in the following:
- Had to reset cluster password (resolved with support)
- No https connection (resolved with support)
- After downloading PlacesAudio, I found out that I had to download them from the original dataset (managed with openxlab)
- Only need 400k images, need to check if I got all of them (being done at the job 8656)
- In the cluster I only have modules cuda 11.8 and 12.1 which they will lead to many problems due to cuda 11.0 code based

Now I proceed to study how to achieve the temporal variable for SSL-TIE

### 06/02/2025
Since the code is based on pytorch 1.12.1 Cuda 11.0, Xavi suggested that instead to migrating all the code try first doing a test:
- I tried to create de conda env with for pytorch 1.12.1 but it was impossible since CUDA 11.8 cannot handle it. the function `torch.cuda.is_available()`  raises an error.
- I created a new environment called `ssl-118` that has pytorch 2.0.0 and CUDA 11.8 and it works. Now `torch.cuda.is_available()` returns `False`. It is expected to return `True`  once I submit the job to the cluster.

For the testing
- I managed to download the model of ssl-tie and the dataset of Flicrk10k and organised it as it required.

Important to mention:
- Now I can use all the Linux lines of code that were hidden since before it was impossible to run them in Windows. Therefore, a change of code is expected.

The test went all right with no issues however we still have the same difference with respect to the paper

### 07/02/2025
Things I could do today:
1. Read to understand how to get the temporal volume
    1. Read Mit paper
    2. Read SSL-TIE
    3. Read DenseAV
    4. See code MIT
    5. Understand and Study the code of the model SSL-TIE (AVE-NET)
    6. Understand the training code
    7. Implement the code
    8. Document everything
    
1.1. My assumption is that this 3rd order tensor M is the final computation and contains all the similarity between the spatial location and temporal of the audio 14 x 14 x 128. They expect these dimensions to be at least recoverable to detect a decent object size and a to at least catching a words within each activation.

It is interesting to see how the conv5 is really important, firstly noted by Zhou et al (Learning deep features for discriminative
localization.) What i think about conv5 is 5x5 filter. They note that they don't couple conv5 to fc1 because that it's a flattening operation meaning that in this way they are able to recover the associations between neurons above and the localisation stimulus that is responsible for the output.

Many interesting techniques to address the problem are described in this paper

- Vision enconder: (14 x 14 x 512/1024)  they apply a 3x3 linear convolution to get the 1024 chanel
- Audio encoder:    (128 x 1024) 128 for the temporal


### 11/02/2025
1.2 Reading SSL-TIE
Remember Siamese Network:
    - Two identical branches with audio and image encoders
    - They use Chen [6] treating the foreground as a negative pair plus unpaired signals
    - Here the shape of the video encoder (14 x 14 x 512) and the shape of audio features(1 x 512)

Purpose of using SSL-TIE:
They key thing of this is that their approach of invariance and equivariance (data augmentation) has been proved that has a audiovisual understanding performing best at Sound Source Localisation
Therefore, in this thesis by means of this approach we try to align the model to Language Spoken modality as done by Hamilton et al. (2024) but with this architecture.
However, since SSL-TIE detects the SS by processing the 10 secs of audio entirely, its audio encoder architecture has to be changed in order to get the volume of similarities.
propose to change the audio feature:
At the end they use
    
### 12/02/2025
AudioConvNet DAVENet
```python
 def __init__(self, embedding_dim=1024):
        super(Davenet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1) # One channel input BW
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0)) # 
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))

    def forward(self, x):
        if x.dim() == 3: # It has to be bw
            x = x.unsqueeze(1) 

        x = self.batchnorm1(x)      # 1024 x 40 x 1
        x = F.relu(self.conv1(x))   # 1024 x 1 x 128
        x = F.relu(self.conv2(x))   # 1024 x 1 x 256 (kernel 1x11 + padding 0x5)
        x = self.pool(x)            # 512 x 1 x 256
        x = F.relu(self.conv3(x))   # 512 x 1 x 512 (kernel 1x17 + padding 0x8)
        x = self.pool(x)            # 256 x 1 x 512
        x = F.relu(self.conv4(x))   # 256 x 1 x 512 (kernel 1x17 + padding 0x8)
        x = self.pool(x)            # 128 x 1 x 512
        x = F.relu(self.conv5(x))   # 128 x 1 x 1024 (kernel 1x17 + padding 0x8)
        x = self.pool(x)            # 64 x 1 x 1024 It should be 128 This maxpooling shouldn't exist
        x = x.squeeze(2)
        return x

```
 the convolution in your code is done with a non-square filter. The kernel size (40, 1) indicates that the filter has a height of 40 and a width of 1. This means the convolution will apply a filter that is tall and narrow, which can be useful for certain types of data, such as audio spectrograms where you might want to capture patterns across frequency bands (height) while preserving the time dimension (width).

### 13/02/2025
So basically this day I managed to:
- reorder my data and put everything at `$DATA`
- Install conda at my laptop
- Create environment to use pytorch in my laptop for later debugging
- Modify code in order to use either cuda environment or cpu
- Check the modification by launching a test with Flickr10k (It passed)
- Start the preparation for debugging the code (``debug_training.sh``)
- We manage to install all the packages correctedly and now we know that we can debug till the breakpoint

Things for tomorrow:
- Make the code possible to use only just a sample in order to train the model
- Check the pipeline
- Decide on the average pooling
- Check what else in the meeting notes

### 15/02/2025
Today we are learning on how to debug

Things to note:
- we may understand later what are the purpose of args.eval_start and args.eval_freq and model_without_dp.state_dict()

Things done:
- Create Launch.json in order to debug the code with the desired parameters
- We understood how the pipeline would go for training by debugging the code
- Now introducing a new dataset_mode called debug
- Upload some sample images and respective audios for dataset debug
- Thanks to debugging, some parameters the model `img_aug` and `aud_aug` are now described in opts file

### 16/02/2025
So today finally we ended debbugging the whole pipeline meaning that
1. We added correctly the `dataset_mode` to `Debug` correctly
2. We wrote down the whole pipeline of the AVENet at the notebook (Todo Daily), the image and the aud
    the audio is worth studying
3. We noted down at ``notes_about_opts.md`` glimpses of aud_aug for the future
4. ``model.py`` required some changes in order to work with CPU
5. We added at the ``launch.json``
6. We also added args.testset_path explained at notes_about_opts.md and the txt for the Dataloader in case of using dir_for_debugging

Things important to note:
1. We wrote down when the model averages the temporal dimension and how it joins both embeddings [torch.einsum, line 284](./models/model.py#L76) 
2. The pipeline works presumably well. However, at training, once the heatmap is forwarded we catch an Exception when the code tries to calculate `calc_topk_accuracy`
```python
top1, top5 = calc_topk_accuracy(out, target, (1,5))
```
[train_one_epoch( ), main.py](main.py#144) Which the error says that index went out of range, which I assume that it is due to having only 5 samples in the dataset and that this is probably the adding thing of negative samples.

3. The testing with `Debug` would work well eventually





