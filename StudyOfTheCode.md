# Study of the code
Here we study the code of the model SSL-TIE. We start with the `main.py` file in order to run the model and see if it works. The procedure will be to run the model with the provided weights and data; maybe we test it and provide the results to verify that the model is working as expected.

## MAIN main.py
directly to def main() function, [`main.py`, line 427](./main.py#L427)

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

It is important to note that the code assumes an evironment variable `CUDA_VISIBLE_DEVICES` to be set. If it is not set, it will raise an error to set it to `args.gpus`, changed at the commit of [`Edit to handle the gpu compability`](https://github.com/jinxiang-liu/SSL-TIE/commit/9d753366b3dc5e3b16cb3169fbc185ae3a1416ac) alongside with some modifications to ensure the initialization of Cuda context.