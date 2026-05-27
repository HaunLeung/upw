# mixed files format
Mixed Image And Text Unsupervised Pretraining requires special mixed files format. 

The text file contains sentence paragraphs and image references. And request to image referenced using <|image|> and <|/image|> symbols. For example: 
```python
Provide a brief description of the given image. <|image|>GCC_train_002582585.jpg<|/image|> olive oil is a healthy ingredient used liberally .
```

We support multiple images references in one mixed file. UPWDataset will automatic reading and segmentation.
```python
We picked a lot of flowers in the garden. 
<|image|>xxxxx_1.jpg<|/image|> 
<|image|>xxxxx_2.jpg<|/image|> 
<|image|>xxxxx_3.jpg<|/image|> 
The first picture is a rose, the second picture is a peony, and the third picture is a lily.
```

These are three mixed files exmaples.