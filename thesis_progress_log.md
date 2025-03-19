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

### 17/02/2025
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

### 19/02/2025
After reading what we did the other day, we have few things that we could start from. I had an interview that really got me unmindful So what i know for sure is the following:

1. We did an analysis on how the input audio is transformed into spectrogram. In which the conclusion is that SSL-TIE respects the temporal resolution while having high frequency resolution compared to DenseAV and DAVENet.

2. SSL-TIE doesn't fully stack the frequential and does reduces the Temporal two times more than DAVENet

[CHECK word ``AUDIO ENCODER INFORMATION``]

Suggestion to do:

1. Do another main that doesn't use the masks 

2. Suggest that maybe  avoiding  another maxpool could lead to better temporal activations

TO DO:

1. Check Harwath how does this the joining 
2. Start by creating a more functional main. Maybe we have to see if we need to restructure the code to make it more reusable by means of encapsulation 
3. Prepare the whole dataset
4. Study how to implement Wandb here. Maybe ask Xavi
5. [If possible] Launch one epoch to inform the whole dataset process.

### 20/02/2025

Acording to the To Do of yesterday:
1. Harwarth computes the triplet margin ranking loss for each anchor image/caption pair and the impostor image/caption is randomly sampled from the minibatch. [Link](../DAVEnet-pytorch/steps/util.py#88)

Then it uses ``torch.mm`` to join both embedding [Link](../DAVEnet-pytorch/steps/util.py#62)

2. We start by adding a parameter for requesting the three order tensor in opts.py


Things to note:
The logits contains the similarity score between the positive regions and negative ones.
The target in this case should be zero because there should be no correlation.
That's why the criterion is nn.CrossEntropyLoss().
Tbh I can't see if they use negative audios for the cl_loss

Maybe we should propose a new framework for the Siamese concept of equivariance.


Therefore I proceed to calculate the same loss as DAVENet:

Finished for today but in the bus im adding the PlacesAudio dataset:
1. The model has been modified to output only the audio and video feature maps (it doesn't compute the mask). The option --order_3_tensor triggers this.
2. A LinearConv has been added to increase the embd dimensions from 512 to 1024 [link model.py](./models/model.py#L124)
3. main.py has been also modified to do epochs with the matchmap approach
4. This new trainer can avoid the siamese equivariance configuration by not putting the opt --siamese
5. Now that the model computes the matchmap it will compute its loss by the triple marging ranking loss
6. We added at util.py the functions and approaches by Harwath et. al 2018
7. In opts it has been added the following: --simtype (MISA | SISA | SIMA) --order_3_tensor and --siamese
8. launch.json has also this new config
9. A debugging has been completed and fulfills the following requirements:
    1. The output dimensions of the matchmap is 14 x 14 x 58 considering and embedding dimension of 1024
    2. The loss has been correctly computed
    3. All the epochs are correctly done
    4. It catches an exception when it wants to do the validation

Things for next time:
1. Maybe we have to consider a new concept of equivariance: transformations for 3D Tensors??
2. Now we have to add the Places Audio dataset in the code 
3. Maybe we have to propose the validation of Harwath but not that frequently
4. Check the Spec aug!!!!

Questions for tomorrow:
- Is it true that they apply all of this transformations but is not meant as data augmentation? Becaues they never use the original samples

### 21/02/2025
Today we had reunion and everything is written down at Notes - TFG Albert Santos

Task of today 
- maybe do the toy example
DONE

### 25/02/2025
So today we started by checking again the toy example therefore we proceed to study how to implement the PlacesAudio Dataset:
- We took a look on how DenseAV manages this:
    1. It uses a abstract class to define the main properties
    2. It has an extensive control on the path settings and manages metadata maybe for each instance with a df. It is not worth to check
    3. However we could apply the same idea in order to manage better future datasets each one using the same data augmentation propierties.
- I would like to check the propierties and statistics of places audio because maybe we don't have to crop from the center like Harwath does...
- Now put ffmpeg in conda to get the statistics
1. Install ffmpeg locally in the env of 'ssl-cpu'
2. After checking the stats of for example flickr we notice that they are 256x256. SSL-TIE inorder to tackle this it uses a ``RandomResizeCrop`` to 224 if the img_aug is specified, if not a normal `Resize(224, Image.BICUBIC)` so it always uses this cropping. [Reference(dataloader)](./datasets/dataloader.py#L204)
3. We checked the statistics PlacesAudio with ffmpeg at the cluster. Things to note:
    1. fs 16kz
    2. Images are 256x256

Now let us do the abstract class for AVDataset:

Okay so this abstract it only initializes the init and the desired transforms. The get item will stay with the minor change of retrieving the files by other methods. this other methods are abstract and they have to be overriden.
the methods are:
- `__len__`
- `_get_file(self,index)`
- `_get_video(self,file)`
- `_get_audio(self,file)`

Therefore the PlacesAudio class is a child class from AVDAtaset and it retrieves the data by parsing the json file, and all of this is thanks to overriding these functions. The desired split for PlacesAudio it comes with the json file.

### 26/02/2025
- We corrected the use of CUDA and added a handler that if using gpu is desired and cuda is  not enabled it stops the execution

- We finally managed to do one epoch and it lasted 26 minuts

Now we plan to implement WandB:
1. Put all the loggers correctly [X]
2. Save the model correctly [x]
    What I would like is that all the models are saved in `$HOME/models/exp_name/...` And exp_name should exist at CLI trigger
2. Do a mini dataset 
3. Check some epochs
4. implement the images and videos
5. Launch training!!
6. 

add the following:
```python
    import json

    json_file = "train.json"

    with open(json_file, 'r') as f:
        data_json = json.load(f)

    data = data_json['data']
    image_base_path = data_json['image_base_path']
    audio_base_path = data_json['audio_base_path']
    print(image_base_path, audio_base_path)
    print(len(data))

    print(data[0]['image'])

    import random
    # Create a new json file
    new_json_file = "train_subset.json"
    # pick randomnly 320 samples
    subset = random.sample(data, 20)
    # write the subset to the new json file
    dict_json = {'image_base_path': image_base_path, 'audio_base_path': audio_base_path, 'data': subset}
    with open(new_json_file, 'w') as f:
        json.dump(dict_json, f)

    with open(new_json_file, 'r') as f:
        data_json = json.load(f)

    data_n = data_json
    print(data_n)
    print(data_n['data'][0]['image'])

```


### 27/02/2025

- We managed to implement WandB vaguely
- We reconfigured the way to save models
- It is expected to work correctly, thus I proceed to create a mini_subset to check the logs

MAP:
1. Do the subset [X]
   Check args.iteration[x]
   
2. Launch to the cluster  [x]
    1. see WandB procedural [x]
    2. check if the models are being saved [x]
    3. 
3. Check if I can resume training 
3. Put the images inferences somehow
4. Read to put the benchmarks as Harwarth


Notes: 
- In test they make the visualisation of the heatmap check [vis_heatmap_bbox()](./utils/util.py#L37)


Questions for the reunion:
- Shall I save the model every epoch?
- What is the jobs folder used for? And diff between run (meant for sbatch files)
- The name of the run can it be set manually?


This week progress:
1. Vaig fer un Toy Example i funciona esperadament 
    (21/02)

2. Donat que al repo de SSL-TIE té un control dels datasets horrible ( molts ifs)  doncs fixant-me en DenseAV he fet una abstract class de AVDataset.
    On fa totes les tranformacions de SSL-TIE i respecta clarament el processament de les dades.
    gràcies això podrem fer coses més naturalment com DenseAV de juntar diferent datasets, també respecta tot el que había abans
    Tot aixo debuguejat i demostrat

3. Since I was not sure of PlacesAudio I did a mini study with ffmpeg to check the dimensions of the image 256x256 i el samplerate 16 fs
    I did it because DAVENet does a center crop of the image and I was quite sckeptical of SSL-TIE which eventually it always does but with a random center.
    With val set it does a Resize to 224 and then a center crop of 224 like why [Reference](./datasets/dataloader.py#L357)

4. Fa tres dies el cluster no em deixaba utilitzar la GPU, no me la detectaba i bueno no he averiguat perquè passava aixó, però tinc ara un codi que si no es detecta i es demana gpu doncs que no entreni el model.
Al final va funcionar i va trigar en fer un epoch amb PlacesAudio 26 minuts i amb 100 epochs trigarà 2 dies. El vaig fer només per un perque el wandb el volia fet abans i totes les coses sobre com guardar els models

5. Doncs amb el repo de Julia i Xavier vaig aconseguir montar el wandb pero no es conectava a l'internet per una tonteria, i buah vaig trigar en veure'l. Jo ja em pensava que era un problema del cluster, tema proxies i tal.

6. Vaig fer un subset per veure com funciona el wandb

7. Vaig arreglar el merder de SSL-TIE que tenen per guardar models

8. Wandb funciona, pero CUDA_ERROR amb el subset a 6 epochs

9. Arreglat, només era que no li agradava el numero de workers

10. VG_SSL-TIE_subset wandb project

11. Ara que tot funciona he llançat el model pero no te encara com veure les imatges i els videos

12. tinc el roadmap començat 

Notes:
- gc per resetejar la memória
- fer proves per millor el batchsize i el num_workers
- mirar bé aixo de la mem de la GPU
- fer un random imatge ini


### 28/02/2025

I start by checking why the loss is not decreasing.

- First thing checked is the audio does not show a weird waveform, maybe normalizing could help
- NORMALIZED, it says that it's already normalized [Reference](./datasets/dataloader.py#L161)

- Spectrogram are dB so we cannot plot them

-Curiosamente en data loader frame.max (2.64) y min-1.4930


### 04/03/2025
Quizas una solución es acabar con los TB pk no sirven de nada y acceden a la memoria 

### 06/03/2025
Okay so the past days it has been very difficult to go on with the thesis due to the necessity of Networking at MWC therefore little progress have been done. 
- A new main called `main_efficient.py` has been created and this one has nothing to do with the original code it directly computes the matchmap and nothing with the TB
- Three new jobs were submitted yesterday and we have the following conclusions:

    `111384` -> ``normald-r_ord_loss`` has ``16 workers``  and batch size `32` 
            DIED at epoch 9 DataLoader starting again and system memory exploded to 11.57 (maybe GB)
            T-epoch. 1025 secs

    `111409` -> `n_w_4` has 4 workers batch size of 32
            Still running
            Somehow at almost 14 hours (48th epoch) the system memory fell drastically and curiosly it started reading from disk ¿?
            T-epoch 1110 secs aprox 18mins expected finishing time: 2 days and 6 hours
            The GPU memory used in this case is 9% we could double the batch size
            COMPLETED!!!!!!

    `111413` -> `mem_eff` 16 workers and batch size of 32
            DIED at epoch 3
            it had a sudden spike of in system memory 

    `120216`-> `n4_b64` 4 workers and batch size of 64 
            Still running

    `120214` -> `again_mem_eff` n16 b 32
            DIED at Wandb HOWEVER still running.

    `120221` -> `SISA_n4b64`
            Still running

    `120286` -> `n4_b128`
    Code is not available at the moment

    ### 07/03/2025

    Meeting today:
    - MWC coudlnt do much
    - He encontrado el problema y no la solución del loss
    - Input images as shown (Transformations are fine)
    - Ja tinc suposadament montat la pujada de videos al wandb i l'avaluació quantitativa de Harwarth (no té misteri es copiar i pegar(no posat en practica
    ))
    - Did another main  mas limpio because supose que el TB afectaba la memoria
    - He posat la possibilitat de netejar: 
        - la cuda cache dins del batch loader amb la desitjada ``frequencia`` i una vegada acabat el epoch, prime gc i despres cuda cache
    - No té un impacte significant, simplement pot evitar un increment de memoria inesperat el que sí acaba es amb la chain de Wandb
    - Si peta es pk ha passat algo amb el multiprocessing de python
    - Amb 16 workers sempre peta, només s'ha salvat un que li ha petat el wandb
    - Recomanació (n_cpus/2 or nGPUs*4) llavors amb 4 m'ha arribat fins al final
    - Faig proves amb n4 b 64 va bé  nomes incrementa l'us de la GPU pero no hiha significativament una reducció de temps
    
    - Proves que he fet per comprobar la baixada del loss:
        - Les imatges no so son negres
        - L'audio i els spectrogrames son aparentment coherents (freq bands cancelled maybe)
        - El loss es una mica especial peró te gradient i es un valor.

        - Els parametres requereixen gradient peró no es guarden llavors loss.backwards() No funciona

        - No sé pq pero jo el que faig es trobar on es trenca la cadena amb
            ```python
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(f"{name} has gradient computed: {param.grad is not None}")
            ```
        
            Pregunta, només fent una operació ja està en el graf, no?

    El nombre de workers afecta al començament del epoch, es tot el process


- Comentar a Xavi el InfoNCE loss

Feedback:
- Començo de nou el codi
- Aplico InfoNCE
- Connectar els grafos


### 12/03/2025

Today we launched the official code of SSL-TIE with PlacesAudio, apparently is learning something:
- We cloned the offical code
- We added WandB  and places audio dataset and modified to not do the validatons with anno
- It has been submitted to the cluster, now it's apparently learning

With this we confirm that the modifications to fit PlacesAudio do not prevent the model from learning

Now we proceed to apply loss made by Hamilton et al. 2024

pos
tensor([[8.9156e+03, 1.0000e+00, 1.0000e+00],
        [1.0000e+00, 7.5594e+03, 1.0000e+00],
        [1.0000e+00, 1.0000e+00, 8.2540e+03]], grad_fn=<ExpBackward0>)
neg
tensor([[1.0000e+00, 1.0003e+04, 8.2822e+03],
        [7.7864e+03, 1.0000e+00, 6.8114e+03],
        [9.5202e+03, 9.6058e+03, 1.0000e+00]], grad_fn=<ExpBackward0>)


### 14/03/2025

#### REUNIÓ:

1. SSL-TIE de 0:
    - Intentar descarregar Flickr-Sound
        El link cau avegades i no es descarrega complet
        Tot per reproduir els resultats
    - Modificacions:
        - Cluster
        - WandB
        - PlacesAudio Dataset class with the Abstract Class
    
Resultat:
Només entrenat amb PlacesAudio -> apren però lógicament s'estanca (invariant del temps)
Conclusió: La incorporació de PlacesAudio no és el problema

2. InfoNCE Loss (Aplicar Hamilton et al. 2024)

    1. El codi de Hamilton està a lightning, no consegueixo veure d'on surt les similarities 
        [Ref at init](DenseAV/denseav/train.py#L325)
        [Contrastive loss](DenseAV/denseav/train.py#L515)
        [Loss](DenseAV/denseav/train.py#L556)
        Es dedicar massa temps
    2. Implemento bassant-me al paper:
        Faig MISA pels B^2 samples, iterant pel batch dues vegades
Codi [Codi](./utils/util.py#L433)

```python
def infoNCE_loss(image_outputs, audio_outputs,args):
    """
        images_outputs (B x H x W x C) 
        audio_outputs  (B x T x C)
        Assumption: the channel dimension is already normalized
    """
    #gradient of image_outputs = grad_fn=<PermuteBackward0>
    B = image_outputs.size(0)
    device = image_outputs.device
    #TODO: Should we require grad to sims?
    sims = torch.zeros(B, B, device=device)
    mask = torch.eye(B, device=device)

    for i in range(B):
        for j in range(B):
            sim_i_j = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[j]), args.simtype)
            #sim_i_j grad_fn=<MeanBackward0>
            sims[i, j] = sim_i_j
    # sims grad_fn=<CopySlices>
    sims = torch.exp(sims / args.temperature) #0.07
    pos= sims * mask
    neg = sims * (1 - mask)

    # TODO: Normalize the rows and columns???
    # pos = pos / pos.sum(1, keepdim=True)
    # neg = neg / neg.sum(1, keepdim=True)

    # This iterates the images against their negative audios...
    loss_v_a = -torch.log(pos.sum(dim=1) / (pos.sum(dim=1) + neg.sum(dim=1))).mean() / 2 
    # This iterates the audios against their negative images...
    loss_a_v = -torch.log(pos.sum(dim=0) / (pos.sum(dim=0) + neg.sum(dim=0))).mean() / 2
    loss = loss_v_a + loss_a_v

    return loss

def computeMatchmap(I, A):
    """
    Computes the 3rd order tensor of matchmap between image and audio.
    Its the dot product.
    """
    assert(I.dim() == 3)
    assert(A.dim() == 2)  
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
```

El model no apren encara, els epochs ara trigan una hora.

### 18/03
The past few days I was developing the idea of the similarity matrix by means of outerproduct that were eased by torch.einsum. After proving a toy example the model has been is now implementing the InfoNCE loss.

Unfortunately, the model is still not learning. We proceed to study other type of solutions:
- `learning rate`
    SSL-TIE has ``1e-4`` by default and DAVEnet has ``0.001``

### 19/03
Aparently it learns even less with a bigger learning rate.
Let's try with a smaller one:
- Job `129726` lr ``1e-5`` It significant learned something but not considerable yet
- Job `129822` lr `1e-6` It takes to much time for an epoch somehow

- `grads`
    I examined the grads after lossbackward in both models. It turns out that the only difference are the new added layers, which they indeed have also gradient. To understand this please go the jobs folder at the cluster.
    Difference
    L#135 the following module.linearConv.weight has parameter
    L#136 the following module.linearConv.bias has parameter
- `init param`
    SSL-TIE does a weight Kaiming initialization which it has been proved by the job `19926` that works as expected. Meaning that it includes our new added layer
- `2layers`
    `129944` After finding out that in our model both encoders were sharing the same linearConvolution, which it makes no sense. Therefore I added a new layer so now they expand their channel dimension independently. 
    I launched two jobs with this new architecture
        1. `129943` with lr `1e-05`
        2. `129944` with lr `1e-04`

