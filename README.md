# CarelessWhisper - Causal Whisper Streaming Model
Causal Whisper Streaming is a fine tuned version of OpenAI Whisper, which can handle causal data and perform real-time transcription. 

[![arXiv](https://img.shields.io/badge/arXiv-2301.12345-b31b1b.svg)](https://arxiv.org/abs/2508.12301)  [![Demo on Hugging Face](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face-blueviolet?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/MLSpeech/CarelessWhisper-causal-streaming)


## 🔧 Setup
We used Python 3.9.16, PyTorch 2.6.0, and PyTorch-Lightning 2.5.0 to train and test our models.
Portions of this code are adapted from [OpenAI's Whisper](https://github.com/openai/whisper).

To set up the project environment using `conda`, follow these steps:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/tomer9080/CarelessWhisper-streaming
   cd CarelessWhisper-streaming
   ```

> 💡 Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed before proceeding.

2. **Create the conda environment**
    ```bash
    conda env create -f environment.yml
    ```

3. **Activate The environment**
    ```bash
    conda activate careless_whisper
    ```

4. **Install the appropriate PyTorch version**  
   Depending on your hardware and CUDA version, install PyTorch by following the instructions at [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).  
   This project was tested with CUDA 12.4, but it should also work with compatible earlier or later versions.
 
After installing all of the dependencies, you can try to run inference.

## 🤖 Available Models
We fine-tuned three different sizes of Whisper, all support english only transcription.
A `large-v2` that was fine tuned on multilingual data is available, and supports English, French, Spanish, German and Portuguese with chunk size of 300 miliseconds.

| Size | Chunk Size [msec] | Multilingual | 
|:----:|:-----------------:|:------------:|
| base | 40, 100, 200, 300 |  N/A         |
| small| 40, 100, 200, 300, 1000| N/A     |
|large-v2| 40, 100, 200, 300, 1000| 300   |


## 🎤 Running Inference
To run inference, download the repo content, and run from the repository root accroding to following sections.

> **Note:** The models are hosted on the [Hugging Face Hub](https://huggingface.co/), which requires an access token.  
> Make sure you are logged in with your token to access the models.

### How to Apply Your Hugging Face 🤗 Access Token

1. **Create a Hugging Face account** (if you don’t have one) at [https://huggingface.co/join](https://huggingface.co/join).

2. **Generate an access token:**
   - Go to your Hugging Face account settings: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click on **"New token"**, give it a name, select the appropriate scopes (usually `read` is enough), and create it.

3. **Login using the Hugging Face CLI:**  
   Install the CLI if you don’t have it:
   ```bash
   pip install huggingface_hub
   ```
   Then login:
   ```bash
   huggingface-cli login
   ```
   Paste your token when prompted.


### 🖥️ CLI Usage
The transcription model is easily activated using the next command:
```bash
# Using a local microphone for streaming transcription, dumping the recording to out.wav
python transcribe.py \
--output_filename out.wav \
--channels 2 \
--model small \ 
--chunk_size 300 \
--device cuda \
--beam_size 5 \
--ca_kv_cache \
```

A simulation of a stream on a wav file is also available:
```bash
# Simulating a stream on a wav file
python transcribe.py \
--model small \
--chunk_size 300 \
--device cuda \
--beam_size 5 \
--ca_kv_cache \
--wav_file /path/to/audio.wav \
--simulate_stream \
--use_latency
```

### 🐍 Python Usage
If you prefer using python, a code sinppet utilizing a microphone or a wav file is provided below:

```python
import torch
import careless_whisper_stream

model_size = "small" # model size
chunk_size = 300 # chunk size in milliseconds
multilingual = False # currently on large-v2_300msec supports other languages than english.
device = "cuda" if torch.cuda.is_available() else "cpu"

model = careless_whisper_stream.load_streaming_model(name=model_size,
                                                   gran=chunk_size,
                                                   multilingual=multilingual,
                                                   device=device)

# using a local microphone recording 
texts_microphone = model.transcribe(output_filename="/path/to/dump/file.wav",
                         channels=2,
                         beam_size=5,
                         ca_kv_cache=True)

# Simulating on a wav file
texts_wav_simulation = model.transcribe(simulate_stream=True,
                                        wav_file="/path/to/file/you/want/to/transcribe.wav",
                                        beam_size=5,
                                        ca_kv_cache=True)
```

## 🦾 Training
In order to train using LoRA, you can use our existing code. Make sure all the requirements are installed. 

### 📂 Dataset Structure

Before starting model training using the command-line interface provided below, you must first configure your dataset dictionary file located at `training_code/ds_dict.py`.

This file defines a Python dictionary named `ds_paths`, where you should specify paths to the `train`, `val`, and `test` partitions of your dataset. Each partition should be a CSV file with the following three columns:

1. `wav_path` — Path to the WAV audio file.  
2. `tg_path` — Path to the corresponding `.TextGrid` file containing forced alignment.  
3. `raw_text` — Ground truth transcription.

> **Note:** The dictionary key (i.e., the name of the dataset) will be used by the training script to identify and load the dataset correctly.

You can find an example entry in `training_code/ds_dict.py`.

### 🖥️ CLI Interface
```bash
python training_code/train.py \
--lora \
--streaming_train \
--simulate_stream \
--dataset LIBRI-960-ALIGNED \
--name example_training_base_model \
--size base \
--batch_size 32 \
--epochs 10 \
--learning_rate 1e-5 \
--rank 32 \
--gran 15 \
--extra_gran_blocks 1 \
--streaming_fraction 0.25 \
--top_k 5 \
```

For more options and training configurations, run:
```bash
python training_code/train.py --help
```

## 📜 License

This repository uses a dual license:

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  
Portions derived from [OpenAI Whisper](https://github.com/openai/whisper) are licensed under the **MIT License**.  

[![CC BY-NC 4.0 License](https://img.shields.io/badge/License-CC--BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)  
All other original code in this repository is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.  

See the [LICENSE](./LICENSE) file for full details.
