<h1 align='center'>Are Transformers Effective for Time Series Forecasting?</h1>

# [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/pdf/2205.13504.pdf)
This is a warehouse for DLinear-Pytorch-model, can be used to train your text dataset for time series forecasting tasks.  
Code mainly from [official source code](https://github.com/cure-lab/LTSF-Linear)

## Preparation
### [Install Packages & Configuration](https://github.com/jiaowoguanren0615/PytorchLearning)
### Download the dataset: 
[TimeSeriesDatasets Nine Benchmarks](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

## Project Structure
```
├── datasets: Load datasets
    ├── data_factory.py: build train/val/test dataset
    ├── mydataset.py: Customize reading data sets and define transforms data enhancement methods
├── models: DLinear Model
    ├── DLinear_model.py: Construct DLinear model
├── util:
    ├── engine.py: Function code for a training/validation process
    ├── losses.py: RMSE Loss
    ├── lr_sched.py: create_lr_scheduler
    ├── masking.py: Construct TriangularCausalMask & ProbMask
    ├── optimizer.py: finetune AdamW optimizer
    ├── samplers.py: Define the parameter of "sampler" in DataLoader
    ├── time_features.py: build time feature data (year, month, hour)
    ├── tools.py: Define help functions (adjust_learning_rate, visual, stanstardscaler, EarlyStopping, etc.)
    ├── utils.py: Record various indicator information and output and distributed environment
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
### Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___, ___num_workers___ and ___nb_classes___ parameters.  


## Train this model

### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error. 

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Citation
```
@inproceedings{Zeng2022AreTE,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Ailing Zeng and Muxi Chen and Lei Zhang and Qiang Xu},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```
