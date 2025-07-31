
# GPU Metric Collection and Fault Tolerance for Megatron-LM

This repository provides a modular framework for simulating GPU failures, collecting detailed GPU metrics, and recovering distributed training jobs—specifically targeting Megatron-LM training pipelines.

---

## 📁 Project Structure

```
├── gpu_metric/                        # GPU metric collection and failure simulation
│   ├── gpu_logs/                      # Raw logs directory
│   ├── gpu_detailed_metrics.csv       # Detailed GPU usage data
│   ├── gpu_detailed_metrics_intenseload.csv  # Metrics under high-load scenarios
│   ├── gpu_metrics.csv                # Aggregated GPU metrics
│   ├── checkpointing.py               # Checkpoint save/load utilities
│   ├── gpu_failure_intenseload.py     # Simulate failures under intense GPU usage
│   ├── gpu_failure_simulation.py      # General-purpose GPU failure simulation
│   ├── metric_collection.py           # Scripts to collect GPU health and usage data
│   └── recover.py                     # Logic for recovery after GPU faults
│
├── master_scripts/                    # Orchestration and control
│   ├── master_node.py                 # Central controller for scheduling and coordination
│   ├── test.py                        # Integration test entry point
│   └── recovery_utils.py              # Shared recovery helper functions
│
└── megatron/                          # Megatron-LM training scripts and examples
    ├── training/                      # Core training implementation
    │   └── training.py                # Launch training with fault tolerance hooks
    ├── examples/gpt3/                 # Example GPT-3 configs and helper files
    ├── train_gpt3_175b_distributed.sh # Shell script to train GPT-3 175B model
    └── README.md                      # Megatron-specific documentation
```

Megatron-LM & Megatron-Core
===========================
<h4>GPU optimized techniques for training transformer models at-scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.5.0-green)](./setup.py)
[![license](https://img.shields.io/badge/license-OpenBSD-blue)](./LICENSE)


# Table of Contents

- [Megatron-LM \& Megatron-Core](#megatron-lm--megatron-core)
- [Megatron Overview](#megatron-overview)
  - [Megatron-LM](#megatron-lm)
  - [Megatron-Core](#megatron-core)
- [Training Speed and Scalability](#training-speed-and-scalability)
- [Setup](#setup)
  - [Downloading Checkpoints](#downloading-checkpoints)
- [Usage](#usage)
- [Training](#training)
  - [Data Preprocessing](#data-preprocessing)
  - [GPT-3 Example](#gpt-3-example)

# Megatron Overview
This repository comprises two essential components: **Megatron-LM** and **Megatron-Core**. Megatron-LM serves as a research-oriented framework leveraging Megatron-Core for large language model (LLM) training. Megatron-Core, on the other hand, is a library of GPU optimized training techniques that comes with formal product support including versioned APIs and regular releases. You can use Megatron-Core alongside Megatron-LM or [Nvidia NeMo Framework](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/nemo_megatron/mcore_customization.html) for an end-to-end and cloud-native solution. Alternatively, you can integrate Megatron-Core's building blocks into your preferred training framework.

## Megatron-LM
First introduced in 2019, Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf), [2](https://arxiv.org/pdf/2104.04473.pdf), and [3](https://arxiv.org/pdf/2205.05198)) sparked a wave of innovation in the AI community, enabling researchers and developers to utilize the underpinnings of this library to further LLM advancements. Today, many of the most popular LLM developer frameworks have been inspired by and built directly leveraging the open-source Megatron-LM library, spurring a wave of foundation models and AI startups. Some of the most popular LLM frameworks built on top of Megatron-LM include [Colossal-AI](https://github.com/hpcaitech/ColossalAI), [HuggingFace Accelerate](https://github.com/huggingface/accelerate), and [NVIDIA NeMo Framework](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/). A list of projects that have directly used Megatron can be found [here](#projects-using-megatron).

## Megatron-Core
Megatron-Core is an open-source PyTorch-based library that contains GPU-optimized techniques and cutting-edge system-level optimizations. It abstracts them into composable and modular APIs, allowing full flexibility for developers and model researchers to train custom transformers at-scale on NVIDIA accelerated computing infrastructure. This library is compatible with all NVIDIA Tensor Core GPUs, including FP8 acceleration support for [NVIDIA Hopper architectures](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/). 

Megatron-Core offers core building blocks such as attention mechanisms, transformer blocks and layers, normalization layers, and embedding techniques. Additional functionality like activation recomputation, distributed checkpointing is also natively built-in to the library. The building blocks and functionality are all GPU optimized, and can be built with advanced parallelization strategies for optimal training speed and stability on NVIDIA Accelerated Computing Infrastructure. Another key component of the Megatron-Core library includes advanced model parallelism techniques (tensor, sequence, pipeline, context, and MoE expert parallelism). 

Megatron-Core can be used with [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/), an enterprise-grade AI platform. Alternatively, you can explore Megatron-Core with the native PyTorch training loop [here](https://github.com/NVIDIA/Megatron-LM/tree/main/examples). Visit [Megatron-Core documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html) to learn more.


# Training Speed and Scalability
Our codebase is capable of efficiently training large language models (i.e., models with hundreds of billions of parameters) with both model and data parallelism. To demonstrate how our software scales with multiple GPUs and model sizes, we consider GPT models ranging from 2 billion parameters to 462 billion parameters. All models use a vocabulary size of 131,072 and a sequence length of 4096. We vary hidden size, number of attention heads, and number of layers to arrive at a specific model size. As the model size increases, we also modestly increase batch size. Our experiments use up to 6144 [H100](https://www.nvidia.com/en-us/data-center/h100/) GPUs. We perform fine-grained overlapping of data-parallel (`--overlap-grad-reduce --overlap-param-gather`), tensor-parallel (`--tp-comm-overlap`) and pipeline-parallel communication (enabled by default) with computation to improve scalability. The reported throughputs are measured for end-to-end training and include all operations including data loading, optimizer steps, communication, and even logging. Note that we did not train these models to convergence.

![Model table](images/model_table.png)

Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.

![Weak scaling](images/weak_scaling.png)

We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.

![Strong scaling](images/strong_scaling.png)


# Setup
We strongly recommend using the latest release of [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) with DGX nodes. If you can't use this for some reason, use the latest pytorch, cuda, nccl, and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start) releases.  Data preprocessing requires [NLTK](https://www.nltk.org/install.html), though this is not required for training, evaluation, or downstream tasks.

You can launch an instance of the PyTorch container and mount Megatron, your dataset, and checkpoints with the following Docker commands:
```
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm -v /path/to/megatron:/workspace/megatron -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints nvcr.io/nvidia/pytorch:xx.xx-py3
```

## Downloading Checkpoints
We have provided pretrained [BERT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m) and [GPT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_lm_345m) checkpoints to evaluate or for finetuning downstream tasks. To access these checkpoints, first [sign up](https://ngc.nvidia.com/signup) for and [setup](https://ngc.nvidia.com/setup/installers/cli) the NVIDIA GPU Cloud (NGC) Registry CLI. Further documentation for downloading models can be found in the [NGC documentation](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1).

Alternatively, you can directly download the checkpoints using:

<pre>
BERT-345M-uncased: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip -O megatron_bert_345m_v0.1_uncased.zip
BERT-345M-cased: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O megatron_bert_345m_v0.1_cased.zip
GPT-345M: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
</pre>

The models require vocabulary files to run. The BERT  WordPiece vocab file can be extracted from Google's pretrained BERT models: [uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt), [cased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt). The GPT [vocab file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) and [merge table](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) can be downloaded directly.

# Usage

After installation, there are several possible workflows. The most comprehensive is:
1. Data preprocessing
2. Pretraining
3. Finetuning (Optional for zero-shot tasks)
4. Downstream task evaluation or text generation

However, steps 1 and 2 can be replaced by using one of the pretrained models mentioned above.

We've provided several scripts for pretraining both BERT and GPT in the [`examples`](./examples) directory, as well as scripts for both zero-shot and fine-tuned downstream tasks including MNLI, RACE, WikiText103, and LAMBADA evaluation. There is also a script for GPT interactive text generation.

# Training
## Data Preprocessing
The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap format use `preprocess_data.py`. An example script to prepare data for BERT training is:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-bert \
       --vocab-file bert-vocab.txt \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
</pre>

The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. The `--data-path` specified in later BERT training is the full path and new filename, but without the file extension.

For T5 use the same preprocessing as BERT, perhaps renaming it to:
<pre>
       --output-prefix my-t5 \
</pre>

Some minor modifications are required for GPT data preprocessing, namely, the addition of a merge table, an end-of-document token, removal of sentence splitting, and a change to the tokenizer type:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-gpt2 \
       --vocab-file gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
</pre>

Here the output files are named `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`. As before, in GPT training, use the longer name without the extension as `--data-path`.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).

## GPT Pretraining

The `examples/gpt3/train_gpt3_175b_distributed.sh` script runs single GPU 345M parameter GPT pretraining. As mentioned above, single GPU training is primarily intended for debugging purposes, as the code is optimized for distributed training.

It follows largely the same format as the previous BERT script with a few notable differences: the tokenization scheme used is BPE (which requires a merge table and a `json` vocabulary file) instead of WordPiece, the model architecture allows for longer sequences (note that the max position embedding must be greater than or equal to the maximum sequence length), and the `--lr-decay-style` has been set to cosine decay.  Note that the `--data-path` now includes the additional `_text_document` suffix added in preprocessing, but does not include the file extensions.

Further command line arguments are described in the source file [`arguments.py`](./megatron/training/arguments.py).

`train_gpt3_175b_distributed.sh` can be launched the same way as described for BERT. Set the env vars and make any other modifications, launch the container with appropriate mounts, and run the script.
More details in [`examples/gpt3/README.md`](./examples/gpt3/README.md)

## Distributed Pretraining

The `pretrain_{bert,gpt,t5}_distributed.sh` scripts use the PyTorch distributed launcher for distributed training. As such, multi-node training can be achieved by properly setting environment variables. See the official PyTorch [documentation](https://pytorch.org/docs/stable/elastic/run.html#launcher-api) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default, multi-node training uses the [nccl](https://developer.nvidia.com/nccl) distributed backend. A simple set of additional arguments and the use of the PyTorch distributed module with the `torchrun` elastic launcher (equivalent to `python -m torch.distributed.run`) are the only additional requirements to adopt distributed training. See any of `pretrain_{bert,gpt,t5}_distributed.sh` for more details.

We use two types of parallelism: data and model parallelism. Our data parallelism implementation is in `megatron/core/distributed`, and supports overlapping of the gradient reduction with the backward pass when the `--overlap-grad-reduce` command-line option is used.

Second, we developed a simple and efficient two-dimensional model-parallel approach. To use the first dimension, tensor model parallelism (splitting execution of a single transformer module over multiple GPUs, see Section 3 of [our paper](https://arxiv.org/pdf/1909.08053.pdf)), add the `--tensor-model-parallel-size` flag to specify the number of GPUs among which to split the model, along with the arguments passed to the distributed launcher as mentioned above. To use the second dimension, sequence parallelism, specify `--sequence-parallel`, which also requires tensor model parallelism to be enabled because it splits across the same GPUs (more details in Section 4.2.2 of [our paper](https://arxiv.org/pdf/2205.05198.pdf)).

To use pipeline model parallelism (sharding the transformer modules into stages with an equal number of transformer modules on each stage, and then pipelining execution by breaking the batch into smaller microbatches, see Section 2.2 of [our paper](https://arxiv.org/pdf/2104.04473.pdf)), use the `--pipeline-model-parallel-size` flag to specify the number of stages to split the model into (e.g., splitting a model with 24 transformer layers across 4 stages would mean each stage gets 6 transformer layers each).

We have examples of how to use these two different forms of model parallelism the example scripts ending in `distributed_with_mp.sh`.

Other than these minor changes, the distributed training is identical to the training on a single GPU.

The interleaved pipelining schedule (more details in Section 2.2.2 of [our paper](https://arxiv.org/pdf/2104.04473.pdf)) can be enabled using the `--num-layers-per-virtual-pipeline-stage` argument, which controls the number of transformer layers in a virtual stage (by default with the non-interleaved schedule, each GPU will execute a single virtual stage with `NUM_LAYERS / PIPELINE_MP_SIZE` transformer layers). The total number of layers in the transformer model should be divisible by this argument value. Additionally, the number of microbatches in the pipeline (computed as `GLOBAL_BATCH_SIZE / (DATA_PARALLEL_SIZE * MICRO_BATCH_SIZE)`) should be divisible by the `PIPELINE_MP_SIZE` when using this schedule (this condition is checked in an assertion in the code). The interleaved schedule is not supported for pipelines with 2 stages (`PIPELINE_MP_SIZE=2`).

## Activation Checkpointing and Recomputation

To reduce GPU memory usage when training a large model, we support various forms of activation checkpointing and recomputation. Instead of all activations being stored in memory to be used during backprop, as was traditionally the case in deep learning models, only activations at certain "checkpoints" in the model are retained (or stored) in memory, and the other activations are recomputed on-the-fly when needed for backprop. Note that this kind of checkpointing, *activation* checkpointing, is very different from the checkpointing of model parameters and optimizer state, which is mentioned elsewhere.

We support two levels of recompute granularity: `selective` and `full`. Selective recomputation is the default and is recommended in almost all cases. This mode retains in memory the activations that take less memory storage space and are more expensive to recompute and recomputes the activations that take more memory storage space but are relatively inexpensive to recompute. See [our paper](https://arxiv.org/pdf/2205.05198) for details. You should find that this mode maximizes performance while minimizing the memory required to store activations. To enable selective activation recompute simply use `--recompute-activations`.

For cases where memory is very limited, `full` recompute saves just the inputs to a transformer layer, or a group, or block, of transformer layers, and recomputes everything else. To enable full activation recompute use `--recompute-granularity full`. When using `full` activation recompute, there are two methods: `uniform` and `block`, chosen using the `--recompute-method` argument.

* The `uniform` method uniformly divides the transformer layers into groups of layers (each group of size `--recompute-num-layers`) and stores the input activations of each group in memory. The baseline group size is 1 and, in this case, the input activation of each transformer layer is stored. When the GPU memory is insufficient, increasing the number of layers per group reduces the memory usage, enabling a bigger model to be trained. For example, when `--recompute-num-layers` is set to 4, only the input activation of each group of 4 transformer layers is stored.

* The `block` method recomputes the input activations of a specific number (given by `--recompute-num-layers`) of individual transformer layers per pipeline stage and stores the input activations of the remaining layers in the pipeline stage. Reducing `--recompute-num-layers` results in storing the input activations to more transformer layers, which reduces the activation recomputation required in the backprop, thus improving training performance while increasing memory usage. For example, when we specify 5 layers to recompute of 8 layers per pipeline stage, the input activations of only the first 5 transformer layers are recomputed in the backprop step while the input activations for the final 3 layers are stored. `--recompute-num-layers` can be incrementally increased until the amount of memory storage space required is just small enough to fit in the available memory, thereby both maximally utilizing memory and maximizing performance.

# Evaluation and Tasks

We provide several command line arguments, detailed in the scripts listed below, to handle various zero-shot and fine-tuned downstream tasks. However, you can also finetune your model from a pretrained checkpoint on other corpora as desired. To do so, simply add the `--finetune` flag and adjust the input files and training parameters within the original training script. The iteration count will be reset to zero, and the optimizer and internal state will be reinitialized. If the fine-tuning is interrupted for any reason, be sure to remove the `--finetune` flag before continuing, otherwise the training will start again from the beginning.

Because evaluation requires substantially less memory than training, it may be advantageous to merge a model trained in parallel for use on fewer GPUs in downstream tasks. The following script accomplishes this. This example reads in a GPT model with 4-way tensor and 4-way pipeline model parallelism and writes out a model with 2-way tensor and 2-way pipeline model parallelism.

<pre>
python tools/checkpoint/convert.py \
        --model-type GPT \
        --load-dir checkpoints/gpt3_tp4_pp4 \
        --save-dir checkpoints/gpt3_tp2_pp2 \
        --target-tensor-parallel-size 2 \
        --target-pipeline-parallel-size 2

</pre>

Several downstream tasks are described for both GPT and BERT models below. They can be run in distributed and model parallel modes with the same changes used in the training scripts.

## GPT Evaluation
We include example scripts for GPT evaluation on WikiText perplexity evaluation and LAMBADA Cloze accuracy.

### WikiText Perplexity Evaluation
For even comparison with prior works, we evaluate perplexity on the word-level [WikiText-103 test dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), and appropriately compute perplexity given the change in tokens when using our subword tokenizer.

We use the following command to run WikiText-103 evaluation on a 345M parameter model.
<pre>
TASK="WIKITEXT103"

VALID_DATA=&#60;wikitext path&#62;.txt
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 1024 \
                  --max-position-embeddings 1024 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng
</pre>

# Checkpoint conversion

We support two forms of model conversion:

1. Model class conversion (i.e., the `GPTModel` in `model.legacy` vs. `model.core`)
2. Checkpoint format conversion (i.e., distributed vs. non-distributed checkpoint)

## Model class conversion

Megatron supports converting between different model classes, including internal model classes (we currently have the older `legacy` models, and the newer `core` models) and external model classes (such as Meta, Huggingface, Mistral, and Mixtral models). Additionally, during this conversion, one can update the parallel state of the model (i.e., changing tensor and pipeline model parallelism).

 We provide the tool `tools/checkpoint/convert.py` to convert between model classes. Some important arguments include:

- `--model-type`: `GPT` or `BERT`
- `--loader`: format of the existing checkpoint. Supported formats include:
  - `legacy`: our older model classes (under `megatron.legacy.model`)
  - `core`: our newer model classes (under `megatron.core.models`)
  - `llama_mistral`: for loading Llama and Mistral models (supports Meta and Huggingface formats)
  - `mixtral_hf`: for loading Mixtral models (Huggingface only)
- `--load-dir`: directory for loading the existing checkpoint
- `--saver`: `legacy` or `core` (see descriptions under `--loader`)
- `--save-dir`: directory for saving the new checkpoint
- `--target-tensor-parallel-size`: new tensor model parallel size
- `--target-pipeline-parallel-size`: new pipeline model parallel size

For more argument details, please see the main script (`convert.py`), loader scripts (`loader_core.py`, `loader_legacy.py`, `loader_llama_mistral.py`, `loader_mixtral_hf.py`), or saver scripts (`saver_core.py`, `saver_legacy.py`).

An example command for converting a GPT model from the old format (`legacy`) to the new format (`core`) would look as follows:

```
python tools/checkpoint/convert.py \
>   --model-type GPT \
>   --loader legacy \
>   --load-dir ${LEGACY_FORMAT_DIR} \
>   --saver core \
>   --save-dir ${CORE_FORMAT_DIR} \
>   --target-tensor-parallel-size ${TP} \
>   --target-pipeline-parallel-size ${PP} \
```

For examples of converting Llama/Mistral models into Megatron, please see [here](docs/llama_mistral.md).

## Checkpoint format conversion

Megatron offers multiple checkpoint formats, including:

- `torch`: Basic checkpoint format with sequential read & writes, and is tied to a specific tensor/pipeline model parallel state (TP/PP states, respectively). (While a specific checkpoint is tied to a specific TP/PP state, a checkpoint can still be manually converted via the model class converter described above).
- `torch_dist`: Distributed checkpoint format, for fast parallel reads & writes, and also is parallel state agnostic (i.e., one can load the same checkpoint to different TP/PP setups).

Generally speaking, `torch_dist` is the more modern and recommended checkpoint format due to its speed. However, depending on the use case, it may be desirable to convert between these two formats. To do so, launch your *training* script (e.g., via `pretrain_gpt.py`) as you normally would, but with two additional arguments:

- `--ckpt-convert-format ${FORMAT}`: `${FORMAT}` can be one of `torch` or `torch_dist`, as described above.
- `--ckpt-convert-save ${PATH_TO_SAVE_NEW_FORMAT}`: this path should be different than your existing `--load`/`--save` paths, to avoid overwriting the existing checkpoint. After converting, use this new path for your `--load`/`--save` paths.

The general idea of this checkpoint format converter is that it launches the model just as one normally would for training, but before running any training iterations, it saves to the new checkpoint format, and then exits. It is important to note that all other launch args should remain the same, in order for the system to understand the previous checkpoint format.