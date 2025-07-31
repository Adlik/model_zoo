# Qwen3-32B Quantization  Tutorial

## 1. Model Overview

**Qwen3-32B** has the following features:

- Type: Causal Language Models
- Training Stage: Pretraining & Post-training
- Number of Parameters: 32.8B
- Number of Paramaters (Non-Embedding): 31.2B
- Number of Layers: 64
- Number of Attention Heads (GQA): 64 for Q and 8 for KV
- Context Length: 32,768 natively and [131,072 tokens with YaRN](https://huggingface.co/Qwen/Qwen3-32B-AWQ#processing-long-texts).
- Quantization: AWQ 4-bit

## 2. Quantization

We provide two quantized models, with detailed quantization procedures outlined below.

### 2.1. Calibration Data

Since the official Qwen repository does not provide calibration datasets for quantization, we selected two open-source and publicly available datasets as calibration data for this quantization effort. Below are the details of the two calibration datasets:

Pile: https://huggingface.co/datasets/mit-han-lab/pile-val-backup/blob/main/val.jsonl.zst

Code1080: https://huggingface.co/datasets/code-search-net/code_search_net, code1080 consists of 1080 calibration samples, with 180 randomly selected from each of the six datasets. You can download it directly from https://github.com/Adlik/model_zoo/tree/main/LLM/datasets/code_6in1_1080.jsonl.

### 2.2. Quantization Algorithm

We provide two quantized models that achieve comparable accuracy to bfloat16 (bf16) and demonstrate superior accuracy over the official AWQ model on most benchmarks. Detailed quantization procedures are outlined below.

1、Smooth model

We first apply the AWQ algorithm to smooth the model. In most cases, the AWQ-smoothed model maintains relatively good accuracy.

Here we employ AWQ solely as the first step of quantization, utilizing only the scale and clip processing provided by AWQ while saving the smoothed model as a floating-point model. Below are the test commands using [AutoAWQ](https://github.com/Adlik/AutoAWQ/tree/autoawq_qwen3).

```sh
# pile
CUDA_VISIBLE_DEVICES=0 python3 examples/fakequantize_model.py --model-path {fp_model} --dataset-path val.jsonl.zst --use-datasets --seed 0 --max-samples 128 --max-seq-len 512 --group-size 128 --output-path {smoothed_model_path}

# code1080
CUDA_VISIBLE_DEVICES=0 python3 examples/fakequantize_model.py --model-path {fp_model} --dataset-path code_6in1_1080.jsonl --seed 0 --max-samples 128 --max-seq-len 512 --group-size 128 --output-path {smoothed_model_path}
```

2、GPTQ 

**Step 2**: Process the smoothed model saved from Step 1 using the GPTQ algorithm and save the GPTQ-processed model in floating-point format. While GPTQ generally achieves better quantization accuracy than AWQ, its weight updates may suffer from significant outliers in the input dimension of model weights. These outliers can lead to large reconstruction errors during GPTQ's weight optimization phase, resulting in substantial accuracy degradation. Preprocessing with AWQ smoothing before applying GPTQ demonstrates excellent effectiveness in mitigating this issue.

Here we provide the quantization configurations relevant to the GPTQ algorithm.

```python
bits=4,
group_size=128,
desc_act=False,
sym=False,
true_sequential=True,
damp_percent=0.01,
damp_auto_increment: float = 0.0015,
mse=True,
```

To facilitate testing, we have modified a version of AutoGPTQ to support Qwen3 model quantization. The code is located at [Adlik/AutoGPTQ](https://github.com/Adlik/AutoGPTQ/tree/qwen3_quant). You can directly use the following commands for model evaluation.

```shell
# pile
CUDA_VISIBLE_DEVICES=0 python3 examples/quantization/model_fakequant.py --model-path {smoothed_model_path} --output-path {calib_fp_model} --group-size 128 --seed 0 --max-samples 512 --max-seq-len 8192 --dataset-path val.jsonl.zst --use-datasets

# code1080
CUDA_VISIBLE_DEVICES=0 python3 examples/quantization/model_fakequant.py --model-path {smoothed_model_path} --output-path {calib_fp_model} --group-size 128 --seed 0 --max-samples 512 --max-seq-len 8192 --dataset-path code_6in1_1080.jsonl
```

3、Convert to AWQ format

Given the promising inference performance of the AWQ-Marlin kernel currently available in vLLM, we consider converting the previously quantized models into the AWQ format. We also provide format conversion scripts in [AutoAWQ](https://github.com/Adlik/AutoAWQ/tree/autoawq_qwen3). The usage commands are as follows:

```shell
CUDA_VISIBLE_DEVICES=0 python3 examples/pack_model.py --model-path  {calib_fp_model}  --output-path {packed_model_path}
```

## 3. Evaluation

### 3.1. Environment

- Evaluation tool

evaluation tool: https://github.com/modelscope/evalscope

version: 0.17.0

```shell
git clone https://github.com/modelscope/evalscope.git
git checkout -b v0.17.0 tags/v0.17.0
cd evalscope/
pip install -e .
```

Evalscope supports testing various benchmarks. When using it, you can specify the benchmark name to test in `dataset_args`. If `subset_list` is not specified, all subsets will be tested by default. The benchmark names are determined by the registered names in the adapter scripts provided under each dataset in [evalscope/benchmarks](https://github.com/modelscope/evalscope/blob/v0.17.0/evalscope/benchmarks). For specific usage details, please refer to the test scripts provided in Section 3.2.

- vllm0.8.5

nothink:

```bash
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0,1 vllm serve /model  --gpu-memory-utilization 0.9 --served-model-name Qwen3-32B --trust_remote_code --port 48001 --tensor-parallel-size 2
```

think:

```bash
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0,1 vllm serve /model --gpu-memory-utilization 0.9 --served-model-name Qwen3-32B --trust_remote_code --port 48001 --tensor-parallel-size 2 --enable-reasoning --reasoning-parser deepseek_r1
```

- Sampling Parameters

ref. https://huggingface.co/Qwen/Qwen3-32B#best-practices

nothink:

```python
max_tokens=32768
temperature=0.7
top_p=0.8
top_k=20
n=1
presence_penalty=1.5
```

think:

```python
max_tokens=32768
temperature=0.6
top_p=0.95
top_k=20
n=1
min_p=0
presence_penalty=1.5
```

Note： math500/aime24/aime25: max_tokens=38912

### 3.2. Evalscope Testing Configuration

ref. https://evalscope.readthedocs.io/en/latest/best_practice/qwen3.html

nothink:

```shell
generation_config={
    'max_tokens': 38912, 
    'temperature': 0.7, 
    'top_p': 0.8,
    'top_k': 20,
    'n': 1, 
    'presence_penalty': 1.5,
    'chat_template_kwargs': {'enable_thinking': False} 
},

dataset_args={
        'ceval': {
            'few_shot_num': 0,
        }
    },
```

think:

```shell
generation_config={
    'max_tokens': 38912, 
    'temperature': 0.6, 
    'top_p': 0.95,
    'top_k': 20,
    'n': 1, 
    'min_p': 0,
    'presence_penalty': 1.5,
}

dataset_args={
        'ceval': {
            'few_shot_num': 0,
            'filters': {'remove_until': '</think>'}  # Filter out the content of thinking
        }
    },
```

Testing Script

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen3-32B',
    api_url='http://127.0.0.1:48001/v1/chat/completions',
    eval_type='service',
    datasets=[
        'gpqa',
    ],
    dataset_args={
        'gpqa': {
            "subset_list": [
                "gpqa_diamond"
            ],
            'few_shot_num': 0,
        }
    },
    eval_batch_size=128,
    generation_config={
        'max_tokens': 32768,  # Max number of generated tokens, suggested to set a large value to avoid output truncation
        'temperature': 0.7,  # Sampling temperature (recommended value per Qwen report)
        'top_p': 0.8,  # top-p sampling (recommended value per Qwen report)
        'top_k': 20,  # top-k sampling (recommended value per Qwen report)
        'n': 1,  # Number of replies generated per request
        'presence_penalty' : 1.5,
        'chat_template_kwargs': {'enable_thinking': False}  # close thinking mode
    },
    timeout=60000,  # Timeout
    stream=True,  # Use streaming output
)

run_task(task_cfg=task_cfg)
```

## 4. Performance

### 4.1 Benchmarks

| model\benchmarks           | think/non-think | math_500 | AIME 2024 | AIME 2025 | MMLU-REDUX | GPQA-Diamond | ceval | gsm8k | ifeval | iquiz | trivia_qa | CMMLU | mmlu  | 说明 |
| -------------------------- | --------------- | -------- | --------- | --------- | ---------- | ------------ | ----- | ----- | ------ | ----- | --------- | ----- | ----- | ---- |
| qwen3-32B-BF（paper）      | think           | 97.2     | 81.4      | 72.9      | 90.9       | 68.4         | 87.3  | \     | 85.0   | \     | \         | \     | \     |      |
|                            | non-think       | 88.6     | 31.0      | 20.2      | 85.7       | 54.6         | 83.3  | \     | 83.2   | \     | \         | \     | \     |      |
| qwen3-32B-AWQ（paper）     | think           | \        | 79.4      | \         | 90.8       | 69.0         | \     | \     | \      | \     | \         | \     | \     |      |
|                            | non-think       | \        | \         | \         | 85.6       | 53.1         | \     | \     | \      | \     | \         | \     | \     |      |
| Qwen3-32B-BF（self-test）  | think           | 96.0     | 80.0      | 66.67     | 89.04      | 68.18        | 88.63 | 92.72 | 87.92  | 84.17 | 81.43     | 87.25 | 87.02 |      |
|                            | non-think       | 85.2     | 26.67     | 16.67     | 86.09      | 55.05        | 85.81 | 89.01 | 87.50  | 80.83 | 75.21     | 85.48 | 83.25 |      |
| qwen3-32B-AWQ（self-test） | think           | 95.2     | 76.67     | 73.33     | 89.09      | 67.68        | 88.41 | 92.04 | 85.35  | 80.83 | 79.63     | 86.74 | 86.2  |      |
|                            | non-think       | 83.2     | 36.67     | 13.33     | 86.26      | 56.57        | 85.66 | 87.49 | 86.74  | 79.17 | 73.69     | 84.53 | 82.49 |      |
| Qwen3-32B-AWQ-Pile         | think           | 95.2     | 80.0      | 70.0      | 88.51      | 69.7         | 88.26 | 93.71 | 85.07  | 83.33 | 80.08     | 86.6  | 86.45 |      |
|                            | non-think       | 84.6     | 30.0      | 16.67     | 85.03      | 56.57        | 86.03 | 89.54 | 86.72  | 79.17 | 73.5      | 84.54 | 82.7  |      |
| Qwen3-32B-AWQ-Code1080     | think           | 94.4     | 86.67     | 73.34     | 88.18      | 71.72        | 88.34 | 93.56 | 88.21  | 81.67 | 78.62     | 86.36 | 86.43 |      |
|                            | non-think       | 83.6     | 26.67     | 26.66     | 85.98      | 57.07        | 84.92 | 89.39 | 87.77  | 79.17 | 72.59     | 84.54 | 82.04 |      |

### 4.2 Performance

Beyond our previously open-sourced quantization-related work, we have also released the AutoQuant inference kernel, which delivers superior inference performance compared to the Marlin kernel. The kernel has been integrated into [vLLM_0.8.5](https://github.com/Adlik/vllm/tree/vllm_0.8.5_autoquant), enabling deployment-ready inference with vLLM. Using it is straightforward: simply modify the config.json and set the value of "quant_method "to "autoquant".

```json
"quantization_config": {
    "bits": 4,
    "group_size": 128,
    "modules_to_not_convert": null,
    "quant_method": "autoquant",
    "version": "gemm",
    "zero_point": true
  },
```

We conducted comparative performance testing between the Marlin and AutoQuant kernels using the benchmark scripts provided by vLLM. 

**Test Specifications**:

- Hardware: 2x NVIDIA A100-40GB-PCIe GPUs
- Inference Engine: vLLM==0.8.5
- Model: Qwen3-32B-AWQ

```shell
# throughput
CUDA_VISIBLE_DEVICES=4,5 python3 benchmark_throughput.py --model /bigdata/lyg/model/Qwen3-32B-AWQ --input-len 1024 --output-len 1024 -tp 2 --max-model-len 40960 --num-prompts 100

# latency
CUDA_VISIBLE_DEVICES=4,5 python3 benchmark_latency.py --model /bigdata/lyg/model/Qwen3-32B-AWQ --num-iters-warmup 10  --num-iters 50  --batch-size 16 --input-len 512 --output-len 512 -tp 2
```

Detailed test results are as follows:

- Throughput

| kernel\\(tokens/s) | type   | in/out=512 | in/out=1024 | in/out=2048 | in/out=4096 |
| ------------------ | ------ | ---------- | ----------- | ----------- | ----------- |
| awq_marlin         | total  | 2153.85    | 1875.67     | 1310.74     | 910.41      |
|                    | output | 1046.28    | 910.15      | 638.11      | 438.71      |
| autoquant          | total  | 2453.12    | 2111.43     | 1416.66     | 963.93      |
|                    | output | 1198.05    | 1024.29     | 689.29      | 469.88      |

- Latency(average)

| kernel\second | batch | in/out=128 | in/out=512 | in/out=1024 | in/out=2048 |
| ------------- | ----- | ---------- | ---------- | ----------- | ----------- |
| awq_marlin    | 16    | 2.4654     | 10.1091    | 21.3455     | 47.7168     |
|               | 64    | 4.8633     | 20.8356    | 47.3302     | 170.8086    |
| autoquant     | 16    | 2.3916     | 9.9021     | 21.0006     | 46.9298     |
|               | 64    | 4.7231     | 20.2468    | 46.0811     | 168.4375    |