# NOTES about OPTS
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

