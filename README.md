# UPW (UNIFIED PIX TOKEN AND WORD TOKEN GENERATIVE LANGUAGE MODEL)
Our goal is to solve the limitation of visual understanding in the current state-of-art open source multimodal generative language model. This code repository is an experiment to demonstrate the unified pix token and word token generative language model proposed in the paper [UNIFIED PIX TOKEN AND WORD TOKEN GENERATIVE LANGUAGE MODEL](https://arxiv.org/abs/2605.14028). User can conducte image unsupervised pretraining experiments that include [mixed image and text](#mixed-image-and-text-unsupervised-pretraining) or [image only](#image-only-unsupervised-pretraining).

## Abstract of Paper
Since the emergence of Vision Transformer (ViT), it has been widely used in generative language model and generative visual model. Especially in the current state-of-art open source multimodal models, ViT obtained by CLIP or SigLIP method serves as the vision encoder backbone to help them acquire visual understanding capabilities. But this method leads to limitations in visual understanding for details, such as difficulty in recognizing small text or numbers in images. To address these issues, we propose a new model to unify pix token and word token into the generative language model. The new model also features with each pix of image having its own token embedding, color folding, global conditional attention approximation and image unsupervised pretraining.

* Unified Pix Token And Word Token Model overview.
![Unified Model overview](doc/upw_arch.png)

* Unified Pix Token And Word Token Model Process.
![Unified Model Process](doc/upw_arch_follow.png)

## Current issues: The limitation of visual understanding in multimodal generative language model
The widely used open-source multimodal large language models currently suffer from the limitation of visual understanding. There may be difficulties in understanding the details in the image, such as recognizing small text or numbers; either there may be problems with understanding the images in multiple rounds of conversations, with the initial conversation being understood correctly and the subsequent conversations being misunderstood. 

For example, almost all open-source models have error in identifying license plate number of car. 
![Qwen_35](problem/Qwen_35/P1.png)
[See more error examples](#error-examples). 

(NOTE: The test images used in the error examples are located in our code repository 'problem/image' directory.)

## Experiment How to
### Install dependencies
* install flash-attention-turing\\
```python
	git clone https://github.com/ssiu/flash-attention-turing
	cd /path/to/flash-attention-turing
	pip install torch setuptools ninja wheel
	pip install -v . 
```

* install flash_attention_interface\\
The install package is in our code repository "src/flash_attn_api" directory.
```python
	cd /path/to/src/flash_attn_api
	pip install setuptools
	python setup.py install
```

* install tokenizers\\
Need a higher version, for example version 0.22.2.
```python
pip install tokenizers
```

* install torch\\
You must ensure that torch is installed, for example version 2.10.0+cu128.

### Image Only Unsupervised Pretraining
* prepare training data\\
You can use any image dataset. We support images of any ratio. In paper we use the images of LLaVA-CC3M-Pretrain-595K dataset. You should put the images in a directory. For example: /path/to/image.
```python 
mkdir /path/to/image
cd /path/to/image
wget https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip
unzip images.zip
```

* run training\\
We run the script to conducte image only unsupervised pretraining. For example we use 120000 images to train.
```python
cd /path/to/src/model
!python train_model.py /path/to/image -l 120000 --imageonly
```

### Mixed Image And Text Unsupervised Pretraining
* training data format\\
The text file contains sentence paragraphs and image references. And request to image referenced using <|image|> and <|/image|> symbols. For example: 
```python
Provide a brief description of the given image. <|image|>GCC_train_002582585.jpg<|/image|> olive oil is a healthy ingredient used liberally .
```
See more examples in our code repository "src/mixed_files" directory.

* prepare training data\\
Prepare the text files according to the format and put them in a directory. For example: /path/to/mixed_files. Put the images referenced in another directory. For example: /path/to/image.

* run training\\
We run the script to conducte mixed image and text unsupervised pretraining. For example we use 120000 mixed files to train.
```python
cd /path/to/src/model
!python train_model.py /path/to/mixed_files -l 120000 -i /path/to/image
```

## Contact
If you have any questions, please feel free to submit a GitHub issue or contact haunleung@outlook.com.

## Citation
If you find our code and models useful, please kindly cite the following information.

```python
@misc{leung2026unifiedpixtokenword,
      title={Unified Pix Token And Word Token Generative Language Model}, 
      author={Haun Leung and ZiNan Wang},
      year={2026},
      eprint={2605.14028},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.14028}, 
}
```

## Error examples
Almost all open-source models fail in identifying license plate number of car.
* GLM-5.1 or GLM-5V-Turbo
![GLM_5.1](problem/GLM_51/P1.png)
![GLM_5V_Turbo](problem/GLM_5V_Turbo/P1.png)

* KIMI-2.6-Instant
![KIMI_2.6_Instant](problem/KIMI_26/P1.png)
![KIMI_2.6_Instant](problem/KIMI_26/P4.png)

* Qwen-3.5
![Qwen_35](problem/Qwen_35/P1.png)
![Qwen_35](problem/Qwen_35/P2.png)

* DeepSeek
![DeepSeek](problem/DeepSeek/P1.png)

