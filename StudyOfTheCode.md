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
- Also it is weird that they want use the directory of the checkpoint to name the 

11/12/2024
Next task is to find a way to filter all the warnings better and to put the dataset for testing.

