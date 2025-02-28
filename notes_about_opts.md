# NOTES about OPTS
## exp_name
The function of set_path(args) at main.py will create directory of ckpts of the associated  experiment `exp_name`.
```python
def set_path(args):
    if args.resume: 
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: 
        exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        exp_path = 'ckpts/{args.exp_name}'.format(args=args)
        
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')

    if not os.path.exists(img_path): 
        os.makedirs(img_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)
    return img_path, model_path, exp_path

```
So `exp_name` will be the name of the model and everything. 
(Take this into account when training)

## img_aug & aud_aug
These two variables maybe can be setted in order to take the desired data augmentation
A random piece of code that I found on the run:
```python
 if (self.args.aud_aug=='SpecAug') and (self.mode=='train') and (random.random() < 0.8):
            maskings = nn.Sequential(
                audio_T.TimeMasking(time_mask_param=180),
                audio_T.FrequencyMasking(freq_mask_param=35)
                )
            spectrogram = maskings(spectrogram)

```
[`dataloader.py`, line 284](./datasets/dataloader.py#L284)

## trainset_path & testset_path
Interestingly they have diferent definitions of testset wether it's vggss or flickr. This is important for them since they make a mix of both datasets. However, for the current purpose of only to use Places Audio it leads to have a lot of redundant code, explicitly a lot of different clauses regarding the `dataset_mode`

## args.iterations
this argument is basically an independent counter to now how many iterations (epochs) but it always start from 1 and doesn't take into account how many epochs has already done. This is of course to handle the args.resume
