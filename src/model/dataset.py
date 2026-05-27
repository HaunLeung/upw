import os
import json
from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from functools import partial
from PIL import Image 
from torchvision import transforms 
import torch

from tokenizer import UPWTokenizer

@dataclass
class DataElement_:
    role: Optional[str] = field(default="text")
    content: Optional[str] | Optional[List[int]] | Optional[torch.tensor] = field(default=None)
    
@dataclass
class DataElement:
    role: Optional[str] = field(default="text")
    tokens: Optional[torch.tensor] = field(default=None) 
    image: Optional[torch.tensor] = field(default=None) 
    mask_token: Optional[torch.tensor] = field(default=None)
    mask_window: Optional[torch.tensor] = field(default=None)
    last_token: Optional[torch.tensor] = field(default=None)

class ToFoldColor:
    def __init__(self, fold_factor) -> None:
        if fold_factor not in [1,2,4,8,16,32,64,128,256]:
            raise ValueError("fold factor wrong!")
        self.fold_factor = fold_factor

    def __call__(self, img):
        img = img//self.fold_factor
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()" 

class UPWDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer:UPWTokenizer, data_name='mixed', data_path='./dataset', 
                image_path='./dataset', fold_factor=16, image_size=256, cache_path=None, 
                split_dataset=False, split_length=1000, split_index=0, max_data_length=None):
        '''
        Support mixed image and text pretraining data files. 
        Data files can be xxx.png or xxx.jpg image or can be text file.
        Text file is composed of text and image url, the entire file length can be arbitrary.
        Image url must be warped by tokenizer's image_start_tag and image_end_tag. It can be relative address or absolute address.
        Example: "This is my favorite toy. <|image|>/path/to/image/xxxx.jpg<|/image|> There are two cute labubus in the picture."
       
        data_name: any name of your dataset.
        data_path: file folder that includes .png/.jpg image files or text files.
        image_path: if image url in text file is relative address, we will add this path in front of it.
        fold_factor: color fold factor for image.
        image_size: Supports original image of any height and width ratio. It will be resize to image_size and pad automatically.
        cache_path: cache this dataset for resue.
        split_dataset: split the dataset to small pieces when your cpu memory is not enough.
        split_length: the length of samples in one split.
        split_index: the split index of whole dataset.
        max_data_length: max data samples will load.
        '''
        self.all_data = None 
        self.fold_factor = fold_factor
        self.tokenizer = tokenizer
        self.image_path = image_path
        self.image_size = image_size
        self.fold_factor = fold_factor

        chunk_size = int(self.tokenizer.max_seq_len * 4)

        has_load_checkpoint = False
        if cache_path:
            if not split_dataset:
                save_path = f'{cache_path}/{data_name}_fac{fold_factor}_img{image_size}.pth'
            else:
                save_path = f'{cache_path}/{data_name}_fac{fold_factor}_img{image_size}_split{split_length}_{split_index}.pth'
            if os.path.exists(save_path):
                loaded_checkpoint = torch.load(save_path, weights_only=False)
                loaded_data = loaded_checkpoint['data']
                #cut length
                if max_data_length:
                    self.all_data = loaded_data[:max_data_length] 
                    del loaded_data
                else:
                    self.all_data = loaded_data
                has_load_checkpoint = True
        if not has_load_checkpoint:
            self.all_data = self.read_data(data_path, chunk_size, split_dataset, split_length, split_index, max_data_length)
            if cache_path:
                torch.save({'data': self.all_data}, save_path)
    
    def read_data(self, data_path, chunk_size, split_dataset, split_length, split_index, max_data_length=None):   
        datas = []
        
        files = self.list_and_sort_files(data_path)

        if split_dataset:
            files = files[split_index*split_length:(split_index+1)*split_length]
   
        if max_data_length:
            files = files[:max_data_length]

        for file in files:
            if self.is_valid_image_pillow(file):
                data = self.read_image_data(file)
                if data!=None:
                    datas.append(data)
            else:
                datas_ = self.read_text_data(file, chunk_size) 
                if datas_!=None:
                    datas.extend(datas_)
            
        return datas
    
    def read_image_data(self, file_path:str):
        data = []

        image_start = DataElement_("text", content=self.tokenizer.image_start_tag)
        data.append(image_start)

        image = self.read_image(file_path, self.image_size, self.fold_factor)
        image_ele = DataElement_("image", content=image)
        data.append(image_ele)
        
        image_end = DataElement_("text", content=self.tokenizer.image_end_tag)
        data.append(image_end)

        return data
  
    def read_text_data(self, file_path:str, chunk_size):
        def parse_start_end(text:str):
            find_sentence_symbol = False
            for i in range(len(text)):
                char = text[i]
                if char.isupper() or char == '.' or char == '?' or char == '!' or char == ','  or char == ' ':
                    text = text[i:]
                    find_sentence_symbol = True
                    break
            if not find_sentence_symbol:
                return None
            find_sentence_symbol = False
            for i in range(len(text) - 1, -1, -1):
                char = text[i]
                if char == '.' or char == '?' or char == '!' or char == ','  or char == ' ':
                    text = text[:i+1]
                    find_sentence_symbol = True
                    break
            if not find_sentence_symbol:
                return None
            return text
        
        def parse_chunk(text:str, image_start_tag, image_end_tag, image_size, fold_factor):
            data = []
            current_pos = 0
            tag_start_len = len(image_start_tag)
            #tag_end_len = len(image_end_tag)

            check_first_tag_end = False
            while True:
                position = text.find(image_start_tag, current_pos)
                if position != -1:
                    if not check_first_tag_end:
                        check_first_tag_end = True
                        position_end = text.find(image_end_tag, current_pos, position)
                        if position_end != -1:
                            current_pos = position_end
                    content = text[current_pos : position + tag_start_len]
                    ele = DataElement_('text', content)
                    data.append(ele)
                    current_pos = position + tag_start_len
                    #tag end
                    position = text.find(image_end_tag, current_pos)
                    if position!=-1:
                        content = text[current_pos : position]
                        content = content.strip()
                        image_url = None
                        if self.is_valid_image_pillow(content):
                            image_url = content    
                        elif self.is_valid_image_pillow(self.image_path + '/' + content):
                            image_url = self.image_path + '/' + content
                        if image_url:
                            image = self.read_image(image_url, image_size, fold_factor)
                        if image!=None:
                            ele = DataElement_('image', image)
                            data.append(ele)
                        current_pos = position
                    else:
                        break
                else:
                    content = text[current_pos:]
                    ele = DataElement_('text', content)
                    data.append(ele)   
                    break 
            assert data[len(data)-1].role == 'text', 'parse data file error!'
            return data
        
        datas = []
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:  # eof
                    break
                chunk = parse_start_end(chunk)
                if chunk==None:
                    continue
                data = parse_chunk(chunk, self.tokenizer.image_start_tag, self.tokenizer.image_end_tag, self.image_size, self.fold_factor)
                if data!=None:
                    datas.append(data)
                
        return datas

    def read_image(self, file_path:str, image_size, fold_factor):
        try:
            image_ = Image.open(file_path).convert("RGB")
        except Exception:
            return None
        w, h = image_.size
        if w<image_size and h<image_size:
            print(f'****** WARNING: image size is too small! {file_path} *****')
            del image_
            return None
        
        if h>w:
            resize_height = image_size
            resize_width = int((float(w)/float(h))*resize_height)
        else:   
            resize_width = image_size
            resize_height = int((float(h)/float(w))*resize_width)
           
        transform = transforms.Compose([
            transforms.Resize(size=(resize_height,resize_width)),
            transforms.PILToTensor(),
            ToFoldColor(fold_factor=fold_factor)
        ])
        image = transform(image_)
        
        del transform
        del image_

        return image
    
    def list_and_sort_files(self, directory, sort_by='name'):
        """
        :param directory: 
        :param sort_by:  ('name', 'mtime', 'size')
        """
        path = Path(directory)
    
        # 1. read all files except sub dir
        files = [f for f in path.iterdir() if f.is_file() and not f.name.startswith('.')]
        
        # 2. sort file
        if sort_by == 'name':
            files.sort(key=lambda x: x.name.lower()) 
        elif sort_by == 'mtime':
            files.sort(key=lambda x: x.stat().st_mtime)
        elif sort_by == 'size':
            files.sort(key=lambda x: x.stat().st_size)
        else:
            raise ValueError("unsupport sort by")
        return files

    def is_valid_image_pillow(self, file_path):
        """
        check PNG or JPG image
        """
        if not os.path.isfile(file_path):
            return False
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True    
            #with Image.open(file_path) as img:
            #    return img.format in ['PNG', 'JPEG']       
        except Exception:
            return False

    def __len__(self):
        return len(self.all_data)
        
    def __getitem__(self, idx):
        data = self.all_data[idx]
        data_encode = []
        for i, ele in enumerate(data):
            if ele.role == 'text':
                tokens = self.tokenizer.encode(ele.content, bos=True if i==0 else False, eos=True if i==(len(data)-1) else False)
                e = DataElement(role='text', tokens=torch.LongTensor(tokens))
            elif ele.role == 'image':
                image, mask_token, mask_window, last_token  = self.tokenizer.pad_image(ele.content)
                e = DataElement(role='image', 
                        image=torch.LongTensor(image.to(torch.int64)), 
                        mask_token=torch.LongTensor(mask_token), 
                        last_token=torch.LongTensor(last_token), 
                        mask_window=torch.LongTensor(mask_window))
            else:
                raise ValueError('unsupport role')
            data_encode.append(e)
        return data_encode

def collate_fn_padded(batch, tokenizer:UPWTokenizer):
    """
    pad data of batch 
    """
    max_data_length = 0
    data_legth_arr = []
    for data in batch:
        data_length = 0
        for ele in data:
            if ele.role=='text':
                data_length += len(ele.tokens)
            elif ele.role=='image':
                data_length += (tokenizer.image_size // tokenizer.window_szie)**2
        if data_length > max_data_length:
            max_data_length = data_length
        data_legth_arr.append(data_length)

    if max_data_length > tokenizer.max_seq_len:
        max_data_length = tokenizer.max_seq_len

    #pad or truncate
    for i, data_length in enumerate(data_legth_arr):
        if data_length < max_data_length:
            #pad 
            data = batch[i]
            last_ele = data[len(data)-1]
            if last_ele.role != 'text':
                raise ValueError('logic error')
            pad_tokens = torch.fill(torch.zeros((max_data_length-data_length, ), dtype=last_ele.tokens.dtype), tokenizer.pad_id)
            last_ele.tokens = torch.cat([last_ele.tokens, pad_tokens])
        elif data_length > max_data_length:
            #truncate
            data = batch[i]
            diff = data_length - max_data_length
            remove_index = []
            for j in range(len(data) - 1, -1, -1):
                ele = data[j]
                if ele.role == 'text':
                    if len(ele.tokens) > diff:
                        ele.tokens = ele.tokens[:(len(ele.tokens)-diff)]
                        break
                    else:
                        remove_index.append(j)
                        diff = diff - len(ele.tokens)
                        if diff <=0:
                            break
                elif ele.role == 'image':
                    if len(ele.mask_token) > diff:
                        remove_index.append(j)
                        pad_legnth = len(ele.mask_token) - diff
                        if j <=0:
                            raise ValueError('logic error')
                        last_ele = data[j-1]
                        if last_ele.role != 'text':
                            raise ValueError('logic error')
                        pad_tokens = torch.fill(torch.zeros((pad_legnth, ), dtype=last_ele.tokens.dtype), tokenizer.pad_id)
                        last_ele.tokens = torch.cat([last_ele.tokens, pad_tokens])
                        break
                    else:
                        remove_index.append(j)
                        diff = diff - len(ele.mask_token)
                        if diff <=0:
                            break
            for index in remove_index:
                if 0 <= index < len(data):
                    del data[index]
                            
    #label
    label_arr = []
    for i, data in enumerate(batch):
        label_ = None
        pre_lable_is_image_and_zero = False 
        for j, ele in enumerate(data):
            if ele.role == 'text':
                if j == 0:
                    label_ = ele.tokens[1:].clone()
                else:
                    tokens = ele.tokens.clone()
                    if pre_lable_is_image_and_zero:
                        tokens[0] = tokenizer.pad_id
                    pre_lable_is_image_and_zero = False
                    label_ = torch.cat([label_, tokens])
            elif ele.role == 'image':
                if j == len(data) - 1:
                    raise ValueError('logic error!')
                next_ele = data[j+1]
                if next_ele.role != 'text':
                    raise ValueError('logic error!')
                predict_token = tokenizer.predict_pix_token(ele.image, ele.mask_token, next_ele.tokens[0])
                predict_token = predict_token[:-1]
                if ele.mask_token[-1] == 0:
                    pre_lable_is_image_and_zero = True
                else:
                    pre_lable_is_image_and_zero = False
                label_ = torch.cat([label_, predict_token])
        #PAD ID 
        label_[(label_==tokenizer.pad_id).bool()] = -100
        label_arr.append(label_)  
    label = torch.stack(label_arr)
    
    #remove last token
    for i, data in enumerate(batch):
        last_ele = data[len(data)-1]
        if last_ele.role != 'text':
            raise ValueError('logic error!')
        last_ele.tokens = last_ele.tokens[:-1]

    sample = {
        'data': batch,
        'label': label.detach(),
    }
    return sample 

