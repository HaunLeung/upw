import os
import gc
import shutil
from typing import Optional
from functools import partial
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass, field
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda import max_memory_allocated
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from tokenizer import UPWTokenizer, collate_fn_padded
from dataset import UPWDataset
from model import UPW

@dataclass
class TrainArguments:
    # dataset
    data_name: Optional[str] = field(default="mixed")
    data_path: Optional[str] = field(default="/input/dataset/data")
    image_path: Optional[str] = field(default="/input/dataset/image")
    split_dataset: bool = field(default=False)
    split_length: Optional[int] = field(default=1000)
    split_index: Optional[int] = field(default=0)
    dataset_splits: Optional[int] = field(default=120)
    cache_path: Optional[str] = field(default=None)
    max_data_length: Optional[int] = field(default=None)
    # tokenizer
    tokenizer_path: Optional[str] = field(default="tokenizer.json")
    vocab_size: Optional[int] = field(default=32000) 
    PAD_TOKEN_ID: Optional[int] = field(default=3) 
    IMAGE_START_TOKEN_ID: Optional[int] = field(default=4) 
    IMAGE_END_TOKEN_ID: Optional[int] = field(default=5) 
    # model text 
    dim: Optional[int] = field(default=256)
    heads: Optional[int] = field(default=8) 
    kv_heads: Optional[int] = field(default=4)
    layers: Optional[int] = field(default=6)
    norm_eps: Optional[float] = field(default=1e-05)
    multiple_of: Optional[int] = field(default=128) 
    ffn_dim_multiplier: Optional[float] = field(default=1.5)
    rope_theta: Optional[float] = field(default=500000.0)
    use_scaled_rope: bool = field(default=True)
    dropout: Optional[float] = field(default=0.0)
    max_batch_size: Optional[int] = field(default=4) 
    max_seq_len: Optional[int] = field(default=1024) 
    ###image
    image_layers: Optional[int] = field(default=6)
    fold_factor: Optional[int] = field(default=16) 
    image_size: Optional[int] = field(default=224) 
    window_size: Optional[int] = field(default=16) 
    # training 
    output_dir: Optional[str] = field(default="./output")
    epochs: Optional[int] = field(default=1)
    lr: Optional[float] = field(default=1e-3)
    eta_min: Optional[float] = field(default=0.1)
    weight_decay: Optional[float] = field(default=0.0)
    clip_grad_norm: bool = field(default=True)
    batch_size: Optional[int] = field(default=2)
    train_with_ddp: bool = field(default=True)
    amp_data_type: Optional[str] = field(default="fp16")
    train_seed: Optional[int] = field(default=2026) 
    # checkpoint
    max_save_count: Optional[int] = field(default=10)
    checkpoint: bool = field(default=True)

# dataset
def create_dataset(arg:TrainArguments, tokenizer:UPWTokenizer):
    dataset = UPWDataset(tokenizer=tokenizer, data_name=arg.data_name, data_path=arg.data_path, image_path=arg.image_path,  
        fold_factor=arg.fold_factor, image_size=arg.image_size, cache_path=arg.cache_path, 
        split_dataset=arg.split_dataset, split_length=arg.split_length, split_index=arg.split_index, max_data_length=arg.max_data_length)
    return dataset

def create_dataloader(arg:TrainArguments, tokenizer:UPWTokenizer):
    train_dataset = create_dataset(arg=arg, tokenizer=tokenizer) 
    collate_with_args = partial(collate_fn_padded, tokenizer=tokenizer)
    train_kwargs = {'batch_size': arg.batch_size, 'collate_fn': collate_with_args, 'num_workers': 2, 'shuffle': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    return train_loader, train_dataset

# checkpoint
def save_checkpoint(model, optimizer=None, scheduler=None, scaler=None, epoch=0, dataset_index=0, filename='train_checkpoint.pth'):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else 0,
        'scheduler': scheduler.state_dict() if scheduler else 0,
        'scaler': scaler.state_dict() if scaler else 0,
        'epoch': epoch,  # [0, epochs-1]
        'data': dataset_index  #[0, dataset_splits-1]
    }
    torch.save(checkpoint, filename)
    del checkpoint
    checkpoint = None

def load_checkpoint(model, optimizer=None, scheduler=None, scaler=None, filename='train_checkpoint.pth'):
    start_epoch = 0
    dataset_index = 0
    if not os.path.exists(filename):
        print(f'checkpoint {filename} not exsit!')
        return start_epoch, dataset_index
    map_location = None if torch.cuda.device_count()>0 else torch.device('cpu')
    checkpoint = torch.load(filename, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if scaler:
        scaler.load_state_dict(checkpoint['scaler'])
    start_epoch = checkpoint['epoch']
    dataset_index = checkpoint['data']
    print(f'load checkpoint {filename} from epoch {start_epoch}, data {dataset_index}')
    return start_epoch, dataset_index

# parameters
def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

# memory use
def print_gpu_memory_usage(prefix:str, clear=False):
    memory0 = max_memory_allocated(torch.device('cuda:0'))
    memory1 = max_memory_allocated(torch.device('cuda:1'))
    print(f'{prefix}, GPU 0 memory allocated: {memory0 / 1e9:.2f}G, GPU 1 memory allocated: {memory1 / 1e9:.2f}G')
    if clear:
        torch.cuda.empty_cache()

def force_clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

import psutil
def mem_usage():
    cpu_percent = psutil.cpu_percent(interval=1)  # 返回1秒内的CPU使用率
    print(f"CPU Usage: {cpu_percent}%")

    memory = psutil.virtual_memory()
    memory_percent = memory.percent  # 内存使用率
    print(f"Memory Usage: {memory_percent}%")
  
# loss func
class PANDWCrossEntropyLoss(nn.CrossEntropyLoss): 
    def forward(self, pred, label):
        self.reduction = 'mean'
        loss = super(PANDWCrossEntropyLoss, self).forward(pred.permute(0, 2, 1), label)
        return loss
    
# model
def create_model(arg:TrainArguments):
    model = UPW(dim=arg.dim, 
            heads=arg.heads, 
            kv_heads=arg.kv_heads, 
            layers=arg.layers, 
            vocab_size=arg.vocab_size, 
            norm_eps=arg.norm_eps, 
            multiple_of=arg.multiple_of, 
            ffn_dim_multiplier=arg.ffn_dim_multiplier, 
            rope_theta=arg.rope_theta, 
            use_scaled_rope=arg.use_scaled_rope, 
            dropout=arg.dropout, 
            max_batch_size=arg.max_batch_size, 
            max_seq_len=arg.max_seq_len,
            image_layers=arg.image_layers, 
            fold_factor=arg.fold_factor, 
            image_size=arg.image_size, 
            window_size=arg.window_size,
            PAD_TOKEN_ID=arg.PAD_TOKEN_ID,
            IMAGE_START_TOKEN_ID=arg.IMAGE_START_TOKEN_ID,
            IMAGE_END_TOKEN_ID=arg.IMAGE_END_TOKEN_ID, 
            train_with_ddp=arg.train_with_ddp)     
    return model

def train_epoch(model, devices, train_loader, optimizer, loss_fun, epoch, scaler, clip_grad_norm, amp_data_type, split_index=0, writer:SummaryWriter=None):
    fsdp_loss = [0.0, 0]

    model.train()

    total_batch = len(train_loader)
    mistone = total_batch // 5

    for i, batch in enumerate(train_loader):
        datas = batch['data']
        labels = batch['label']
        for data in datas:
            for ele in data:
                if ele.role == 'text':
                    ele.tokens = ele.tokens.to(devices[0])
                elif ele.role == 'image':
                    ele.image = ele.image.to(devices[0])
                    ele.mask_token = ele.mask_token.to(devices[0])
                    ele.last_token = ele.last_token.to(devices[0])
                    ele.mask_window = ele.mask_window.to(devices[0])
        labels = labels.to(devices[0])
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=amp_data_type):
            output = model(datas)
            loss = loss_fun(output, labels)
            loss = loss.sum()
        scaler.scale(loss).backward() #loss.backward()
        
        if clip_grad_norm:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer) #optimizer.step()
        scaler.update()

        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += 1
      
        #if i%mistone==0 or i==total_batch-1:
        #    print(f'Training epoch: {epoch+1}, dataset split: {split_index}, progress: {i/total_batch:.3f} ')
        
    train_loss = fsdp_loss[0] / fsdp_loss[1]
    print(f"Training epoch: {epoch+1}, dataset split: {split_index}, loss: {train_loss:.4f}")
    writer.add_scalar("loss", train_loss, split_index)

    return train_loss

def train(arg:TrainArguments): 
    random.seed(arg.train_seed)
    np.random.seed(arg.train_seed)
    torch.manual_seed(arg.train_seed)

    if torch.cuda.device_count()>0:
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device('cpu')]
    writer = SummaryWriter(f"{arg.output_dir}/log/train_log")
    
    #create tokenizer
    tokenizer = UPWTokenizer(model_path=arg.tokenizer_path, fold_factor=arg.fold_factor, 
                image_size=arg.image_size, window_szie=arg.window_size, max_seq_len=arg.max_seq_len, vocab_size=arg.vocab_size)
    arg.PAD_TOKEN_ID = tokenizer.pad_id
    arg.IMAGE_START_TOKEN_ID = tokenizer.image_start_id
    arg.IMAGE_END_TOKEN_ID = tokenizer.image_end_id

    # create model
    model = create_model(arg)
    print_trainable_parameters(model)

    model = nn.DataParallel(model, device_ids=devices).to(devices[0])

    #loss funtion
    loss_fun = PANDWCrossEntropyLoss()
    #optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, arg.epochs * arg.dataset_splits, arg.eta_min * arg.lr)
    scaler = torch.amp.GradScaler('cuda')
    amp_data_type = torch.float16 if arg.amp_data_type == 'fp16' else torch.bfloat16

    #load checkpoint
    if arg.checkpoint:
        checkpoint_path = arg.output_dir + "/checkpoint/train_checkpoint.pth"
        start_epoch, dataset_index = load_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, filename=checkpoint_path)
        if not (start_epoch==0 and dataset_index==0):
            dataset_index += 1
            if dataset_index >= arg.dataset_splits:
                dataset_index = 0
                start_epoch += 1

    #train loop
    for epoch in range(start_epoch, arg.epochs):
        #create dataset and dataloader
        for i in tqdm(range(dataset_index, arg.dataset_splits), desc=f'Training epoch {epoch+1}'):
            arg.split_index = i
            train_loader, train_dataset = create_dataloader(arg, tokenizer)
            train_epoch(model, devices, train_loader, optimizer, loss_fun, epoch, scaler, arg.clip_grad_norm, amp_data_type, i, writer)

            del train_loader
            del train_dataset
            train_loader = None
            train_dataset = None
            force_clean_memory()
            
            if (epoch==0 and i==0) or (i>0 and (i+1)%arg.max_save_count==0) or i==(arg.dataset_splits-1):
                if arg.checkpoint:
                    checkpoint_path = arg.output_dir + "/checkpoint"
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    checkpoint_path += "/train_checkpoint.pth"
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, i, checkpoint_path)
                    force_clean_memory()
            scheduler.step()
        dataset_index = 0   
   
def create_dataset_cache(arg):
    tokenizer = UPWTokenizer(model_path=arg.tokenizer_path, fold_factor=arg.fold_factor, 
            image_size=arg.image_size, window_szie=arg.window_size, max_seq_len=arg.max_seq_len, vocab_size=arg.vocab_size)
    
    for i in tqdm(range(arg.dataset_splits), desc="create training dataset cache"):
        arg.split_index = i
        train_dataset = create_dataset(arg, tokenizer)
        del train_dataset

    del tokenizer

def main():
    parser = argparse.ArgumentParser(description='image only pretrain or mixed image and text pretrain')
    parser.add_argument('datapath', help='path to the image files or mixed files directory for pretrain')
    parser.add_argument('--length', '-l', type=int, default=0, help='the maximum files used for pretrain')
    parser.add_argument('--imagepath', '-i', default=None, help='the actual directory of the images referenced in the mixed files')
    parser.add_argument('--imageonly', action='store_true', help='image only pretrain or not')
    parser.add_argument('--tokenizer', '-t', default='tokenizer.json', help='the path of tokenizer file')
    parser.add_argument('--imagesize', '-s', type=int, default=224, help='the image size used for pretrain')
    parser.add_argument('--windowsize', '-w', type=int, default=16, help='the patch size of image also called local window size')
    parser.add_argument('--foldfactor', '-f', type=int, default=16, help='the fold factor of color, one of [1, 2, 4, 8, 16, 32, 64, 128, 256]')
    parser.add_argument('--clear', '-c', action='store_true', help='clear dataset cache')
    parser.add_argument('--batchsize', '-b', type=int, default=0, help='the batch size')
    parser.add_argument('--output', '-o', default='./upw', help='the output directory')

    args = parser.parse_args()

    train_args = TrainArguments()
    #dataset
    train_args.data_name = 'image' if args.imageonly else 'mixed'
    train_args.data_path = args.datapath
    train_args.image_path = args.imagepath

    train_args.cache_path = args.output + "/cache_data"
    if args.clear:
        if os.path.exists(train_args.cache_path):
            shutil.rmtree(train_args.cache_path)
    if not os.path.exists(train_args.cache_path):
        os.makedirs(train_args.cache_path)
    
    train_args.split_dataset = True
    train_args.split_length = 1000
    train_args.split_index = 0

    def count_files(directory):
        path = Path(directory)
        return sum(1 for x in path.rglob("*") if x.is_file())
    total_files = count_files(args.datapath)
    if args.length>0 and args.length<total_files:
        total_files = args.length
    splits = (total_files // train_args.split_length) + (0 if total_files%train_args.split_length==0 else 1)
    train_args.dataset_splits = splits #max dataset split to train in one epoch
    
    # tokenizer
    train_args.tokenizer_path = args.tokenizer
    train_args.vocab_size = 6 if args.imageonly else 32000
    #text
    train_args.dim = 768 
    train_args.layers = 12 if args.imageonly else 10
    train_args.heads = 12
    train_args.kv_heads = 6
    train_args.max_seq_len = 1024
    #image
    train_args.image_layers = 5
    train_args.fold_factor = args.foldfactor 
    if not args.foldfactor in [1,2,4,8,16,32,64,128,256]:
        print('*** args error: foldfactor should be one of [1, 2, 4, 8, 16, 32, 64, 128, 256]! ***')
        return 
    train_args.image_size = args.imagesize 
    train_args.window_size = args.windowsize
    if args.imagesize % args.windowsize != 0:
        print('*** args error: imagesize should be divided by windowsize! ***')
        return 
    #train
    train_args.epochs = 1
    train_args.lr = 0.0006
    train_args.weight_decay = 0.01
    train_args.clip_grad_norm = True
    train_args.batch_size = args.batchsize if args.batchsize>0 else (torch.cuda.device_count() if torch.cuda.device_count() > 0 else 2)
    train_args.train_with_ddp = True
    train_args.output_dir = args.output
    # checkpoint
    train_args.checkpoint = True
    train_args.max_save_count = 10

    print(train_args)
 
    create_dataset_cache(train_args)
    
    train(train_args)
    
        
if __name__ == '__main__':
    '''
    image only:
    !python train_model.py /path/to/dataset/llava-595k -l 100000 -t /path/to/tokenizer/tokenizer.json --imageonly

    mixed image and text:
    !python train_model.py /path/to/dataset/mixed_files -l 100000 -i /path/to/dataset/image -t /path/to/tokenizer/tokenizer.json
    '''
    main()