import numpy as np
import torch

def create_win_index(image_size, kernel_size, step, return_index=True, revert=False):
    if (image_size-kernel_size)%step!=0:
        raise ValueError('image_size and kernel size and step error!')
    if kernel_size!=step:
        raise ValueError('kernel size and step must the same!')
    if return_index:
        index = np.array([], dtype=np.int64)
        row = np.array([i for i in range(0, kernel_size)])
        for i in range(0, image_size, step):
            if i + kernel_size > image_size:
                continue
            for j in range(0, image_size, step):
                if j + kernel_size > image_size:
                    continue
                for k in range(0, kernel_size):
                    e = row + j + k*(image_size) + i*image_size
                    index = np.append(index, e)
        if revert:
            revert_index = np.array([0]*len(index), dtype=np.int64)
            for i in range(len(index)):
                revert_index[index[i]] = i
            index = revert_index
        index = torch.from_numpy(index).to(dtype=torch.int64)
    else:
        index = None
    wins = (image_size-kernel_size)//step + 1
    wins = wins**2
    win_es = kernel_size*kernel_size
    return index, wins, win_es

def get_pix_token(image: torch.tensor, fold_factor: int):
    color_length = (256//fold_factor)
    color_index = torch.LongTensor([1, color_length, color_length**2])
    
    image_ = image.permute(1,2,0)
    # * color_index
    color_index = color_index.to(image_.device)
    image_ = image_ * color_index
    #(H,W,3) -> (H,W)
    image_ = image_.sum(dim=2)
    return image_