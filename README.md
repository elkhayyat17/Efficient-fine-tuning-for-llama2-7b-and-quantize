# 🚀🚀Efficient-fine-tuning-for-llama2-7b-and-quantize-

# 1. **PEFT LLAMA 2 7b**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://github.com/elkhayyat17/Mistral7b_pdf_chatting/blob/main/Chat_with_papers_Mistral_7b.ipynb](https://colab.research.google.com/drive/1WMvmjJEdNxcozAId37XkY753Fq-7L0rt?usp=drive_link) <br>
Fine-tuning a large language model **refers to the process of further training the pre-trained model on a specific task or domain using a smaller dataset.** The initial pre-training phase involves training a language model on a massive corpus of text data to learn general language patterns and representations. Fine-tuning, on the other hand, customizes the model to a specific task or domain by exposing it to task-specific data. By fine-tuning a large language model on a specific task, you leverage the pre-trained knowledge of the model while tailoring it to the nuances and requirements of your target task. This typically allows the model to perform better and achieve higher accuracy on the specific task compared to using the pretrained model by itself for your specific task.<br>
**Quantization-Based Fine-Tuning (QLoRA)🦾🦾**:

- Involves reducing the precision of model parameters (e.g., converting 32-bit floating-point values to 8-bit or 4-bit integers). This reduces the amount of CPU and GPU memory required by either 4x if using 8-bit integers, or 8x if using 4-bit integers.
- Typically, since we're changing the weights to 8 or 4 bit integers, we will lose some precision/performance.
- This can lead to reduced memory usage and faster inference on hardware with reduced precision support.
- Particularly useful when deploying models on resource-constrained devices, such as mobile phones or edge devices.<br>


** We're going to fine-tune using method 3 since we only have access to a single T4 GPU with 15GiB of GPU VRAM on Colab.** <br>


To do this, the new parameters we're introducing are:

- `adapter`: The PEFT method we want to use
- `quantization`: Load the weights in int4 or int8 to reduce memory overhead.
- `trainer`: We enable the `finetune` trainer and can configure a variety of training parameters such as epochs and learning rate.<br>

**Upload Trained Model Artifacts To HuggingFace** 🤗

Now that we have a fine-tuned model, we can export the model weights to HuggingFace hub so we can use them in downstream tasks or in production. Ludwig supports uploading model weights directly to HuggingFace Hub via the `upload` Ludwig command.<br>


# 2. **Run inference using the qlora adapter and base model**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://github.com/elkhayyat17/Mistral7b_pdf_chatting/blob/main/Chat_with_papers_Mistral_7b.ipynb](https://github.com/elkhayyat17/Efficient-fine-tuning-for-llama2-7b-and-quantize/blob/main/2-load-peft-adapter.ipynb))<br>
- load qlora adaptor and base model using peft and bitsandbytes libraries
- perform Prompt engineering for model 

# 3. **merge adapter and quantization(gguf)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/elkhayyat17/Efficient-fine-tuning-for-llama2-7b-and-quantize/blob/main/3-merge%20adapter%20and%20quantization(gguf).ipynb)<br>

- Merge adaptor with base model and quantize it to n-bit-gguf fromat and push them to huggingface

# 4. **load gguf** 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/elkhayyat17/Efficient-fine-tuning-for-llama2-7b-and-quantize/blob/main/4-%20load%20gguf.ipynb) <br>
- load and perform Prompt engineering for model 

  

