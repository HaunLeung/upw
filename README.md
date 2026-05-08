# UPW
Resolve the limitation of image understanding in multimodal LLMs

## Current issues: The limitation of image understanding in multimodal LLMs
The widely used open-source multimodal large language models currently suffer from the limitation of image understanding. There may be difficulties in understanding the details in the image, such as recognizing small text or numbers; either there may be problems with understanding the images in multiple rounds of conversations, with the initial conversation being understood correctly and the subsequent conversations being misunderstood. 

For example, the following error example:
Almost all open-source models have error in identifying license plate number of car.

* GLM-5.1
![GLM_5.1](problem/GLM_51/P1.png)

* GLM-5V-Turbo
![GLM_5V_Turbo](problem/GLM_5V_Turbo/P1.png)

* KIMI-2.6-Instant
![KIMI_2.6_Instant](problem/KIMI_26/P1.png)
![KIMI_2.6_Instant](problem/KIMI_26/P4.png)

* Qwen-3.5
![Qwen_35](problem/Qwen_35/P1.png)
![Qwen_35](problem/Qwen_35/P2.png)

* DeepSeek
![DeepSeek](problem/DeepSeek/P1.png)

The test images used in the above examples are located in the 'problem/image' directory