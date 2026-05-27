import os
from typing import List
import torch
from tokenizers import Tokenizer

from utils import create_win_index


class UPWTokenizer:
    """tokenizing and encoding/decoding text using HuggingFace Tokenizer."""
    def __init__(self, model_path: str, fold_factor: int, image_size: int, window_szie: int, max_seq_len: int, vocab_size:int = None):
        """
        Initializes with a Tokenizer model file.
        Args:  model_path (str): The path to the Tokenizer model file.
        """
        assert os.path.isfile(model_path), model_path
        self.model = Tokenizer.from_file(model_path)
        self.bos_id: int = self.model.token_to_id("<s>") #1
        self.eos_id: int = self.model.token_to_id("</s>") #2
        self.pad_id: int = self.model.token_to_id("<pad>") #3
        self.image_start_id: int = self.model.token_to_id("<|image|>") #4 
        self.image_start_tag: str = '<|image|>' 
        self.image_end_id: int = self.model.token_to_id("<|/image|>") #5
        self.image_end_tag: str = '<|/image|>' 
        ##vocab size is 32000 or we set to 6 for only image pretrain
        self.vocab_size: int = vocab_size if vocab_size else self.model.get_vocab_size() 

        self.fold_factor = fold_factor
        self.image_size = image_size
        self.window_szie = window_szie
        self.max_seq_len = max_seq_len

        color_length = (256//fold_factor)
        self.color_index = torch.LongTensor([1, color_length, color_length**2])
    
        pix_tokens_length = color_length**3 + 1 #add one pad pix token
        self.map_tokens = torch.full((pix_tokens_length,), 0, dtype=torch.long)
        self.init_map_tokens()

        select_index, wins, win_es = create_win_index(image_size, window_szie, window_szie)
        select_index = select_index.reshape(-1, win_es)
        self.select_first_index = select_index[:, 0] #(wins,)
        self.wins = wins
        self.win_es = win_es

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
        Returns: List[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.model.encode(s, add_special_tokens=False)
        ids = t.ids
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, t: List[int], skip_special_tokens=False) -> str:
        """
        Decodes a list of token IDs into a string.
        Args: t (List[int]): The list of token IDs to be decoded.
        Returns: str: The decoded string.
        """
        return self.model.decode(t, skip_special_tokens=skip_special_tokens)

    def init_map_tokens(self):
        total = self.map_tokens.shape[0]
        for i in range(total):
            if i == total-1:
                self.map_tokens[i] = self.pad_id
            else:
                self.map_tokens[i] = self.vocab_size + i

    def pad_image(self, image:torch.tensor):
        '''
        Either W or H is image_size and can be divided by window_size.
        If W is divided by window_size but H not, pad PIX_PAD_TOKEN to H of image.
        If H is divided by window_size but W not, pad PIX_PAD_TOKEN to W of image.
        in: 
            image (3,H,W) 
        out: 
            image (3,image_size,image_size)
            mask_token: (wins,), 1 or 0
            mask_window: (win_es,), 1 or 0, for local window attention, first create wins mask all 1 and set last_token != -1 to mask_window
            last_token: (wins,), -1, or last PIX_PAD_TOKEN position
        '''
        _, h, w = image.shape

        assert w==self.image_size or h==self.image_size

        if h <= w:
            #pad pix pad token
            dif = w - h
            word_pad_token_count = 0
            last_token_index = -1
            if dif>0:
                cl = 256 // self.fold_factor
                if cl==256:
                    raise ValueError(f'***** Error Happen, Unsuporot fold_factor: {self.fold_factor} for uint8 dtype *****')
                pad = torch.tensor([cl, cl-1, cl-1], dtype=image.dtype)
                pad = pad.reshape(3,1,1)
                pad = torch.repeat_interleave(pad, w, dim=2)
                pad = torch.repeat_interleave(pad, dif, dim=1)
                image = torch.cat([image, pad], dim=1)
                #for mask token
                word_pad_token_count = (dif//self.window_szie) * (w//self.window_szie)
                #for last_token
                last_token_index = self.win_es - (dif%self.window_szie) * self.window_szie - 1
            #mask_token
            mask_token = torch.ones((self.wins,), dtype=torch.int64)
            mask_token[self.wins - word_pad_token_count:] = 0
            #last_token
            no_pad_token_count =  dif//self.window_szie + 1 if dif%self.window_szie !=0 else 0
            no_pad_token_count = self.wins - no_pad_token_count * (w//self.window_szie)
            last_token = torch.fill(torch.ones((self.wins,), dtype=torch.int64), -1)
            last_token[no_pad_token_count:] = last_token_index 
            #mask_window
            mask_window = torch.ones((self.win_es,), dtype=torch.int64)
            if last_token_index>=0:
                mask_window[last_token_index+1:] = 0
        else:
            #pad pix pad token
            dif = h - w
            cl = 256 // self.fold_factor
            if cl==256:
                raise ValueError(f'***** Error Happen, Unsuporot fold_factor: {self.fold_factor} for uint8 dtype *****')
            pad = torch.tensor([cl, cl-1, cl-1], dtype=image.dtype)
            pad = pad.reshape(3,1,1)
            pad = torch.repeat_interleave(pad, dif, dim=2)
            pad = torch.repeat_interleave(pad, h, dim=1)
            image = torch.cat([image, pad], dim=2)
            #for last_token
            last_token_index = self.win_es - dif%self.window_szie - 1
            #mask_token
            mask_token = torch.ones((self.wins,), dtype=torch.int64)
            mask_token_index = []
            for i in range(dif//self.window_szie):
                for j in range(h//self.window_szie):
                    index = (j+1) * (h//self.window_szie) - 1 - i
                    mask_token_index.append(index)
            mask_token[mask_token_index] = 0
            #last_token
            pix_pad_token_count = dif//self.window_szie + 1 if dif%self.window_szie !=0 else 0
            last_token = torch.fill(torch.ones((self.wins,), dtype=torch.int64), -1)
            pix_pad_token_index = []
            for i in range(pix_pad_token_count):
                for j in range(h//self.window_szie):
                    index = (j+1) * (h//self.window_szie) - 1 - i
                    pix_pad_token_index.append(index)
            last_token[pix_pad_token_index] = last_token_index
            #mask_window
            mask_window = torch.ones((self.win_es,), dtype=torch.int64)
            mask_window_index = []
            for i in range(dif%self.window_szie):
                for j in range(self.window_szie):
                    index = (j+1) * (self.window_szie) - 1 - i
                    mask_window_index.append(index) 
            mask_window[mask_window_index] = 0

        return image, mask_token, mask_window, last_token

    def predict_pix_token(self, image:torch.tensor, mask_token:torch.tensor, next_token:torch.tensor):
        '''
        return the first predict pix token of local window. If mask_token is 0 return WORD_PAD_TOKEN
        images: (3,image_size,image_size)
        mask_token: (wins,)
        next_token: one element tensor
        out: (wins+1,); p1, pk, p2k, ..., pn, </img>
        '''
        assert next_token.numel() == 1
        next_token = next_token.reshape(-1)

        image = image.permute(1,2,0)
        self.color_index = self.color_index.to(image.device)
        self.map_tokens = self.map_tokens.to(image.device)
        image = image * self.color_index
        image = image.sum(dim=2)
        image = image.reshape(-1)
        image = image[self.select_first_index]
        image = self.map_tokens[image]
        image[~(mask_token.bool())] = self.pad_id
        image = torch.cat([image, next_token])
        mask_one = torch.ones((1,), dtype=torch.int64)
        mask_token_ = torch.cat([mask_one, mask_token])
        mask_token = torch.cat([mask_token, mask_one])
        for j, mask in enumerate(mask_token):
            if mask == 1:
                continue
            else:
                for k, mask_ in enumerate(mask_token):
                    if k<=j:
                        continue
                    if mask_ == 0:
                        continue
                    else:
                        image[j] = image[k]
                        break

        #PAD ID
        image[~(mask_token_.bool())] = self.pad_id
        return image

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
