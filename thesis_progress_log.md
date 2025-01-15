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
