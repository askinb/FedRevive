# FedRevive: Reviving Stale Updates with Data-Free Knowledge Distillation

<p align="center">
  <a href="https://arxiv.org/abs/2511.00655"><img src="https://img.shields.io/badge/arXiv-2511.00655-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/NeurIPS-2026-6f42c1.svg" alt="NeurIPS 2026">
  <a href="https://github.com/NVIDIA/NVFlare/tree/main/research/fedrevive"><img src="https://img.shields.io/badge/NVFlare-implementation-76B900.svg?logo=nvidia&logoColor=white" alt="NVFlare implementation"></a>
</p>

This repository contains the official PyTorch implementation of

> **[Reviving Stale Updates: Data-Free Knowledge Distillation for Asynchronous Federated Learning](https://arxiv.org/abs/2511.00655)**<br>
> Baris Askin<sup>1,†</sup>, Holger R. Roth<sup>2</sup>, Zhenyu Sun<sup>3,†</sup>, Carlee Joe-Wong<sup>1</sup>, Gauri Joshi<sup>1</sup>, Ziyue Xu<sup>2</sup><br>
> <sup>1</sup>Carnegie Mellon University, <sup>2</sup>NVIDIA, <sup>3</sup>Northwestern University<br>
> *The Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS 2026)*<br>
> <sup>†</sup>Work done during an internship at NVIDIA.

📄 **Paper:** https://arxiv.org/abs/2511.00655<br>
<img src="https://cdn.simpleicons.org/nvidia/76B900" width="16" height="16" alt="NVIDIA"> **NVIDIA FLARE implementation:** https://github.com/NVIDIA/NVFlare/tree/main/research/fedrevive

## Abstract

Federated learning (FL) enables collaborative model training across distributed clients without sharing raw data, yet its scalability is limited by synchronization overhead. Asynchronous federated learning (AFL) alleviates this issue by allowing clients to communicate independently, thereby improving wall-clock efficiency in large-scale, hardware-heterogeneous environments. However, asynchrony introduces updates computed on outdated global models (staleness) that can destabilize optimization and hinder convergence. We propose FedRevive, an AFL framework that revives stale updates through data-free knowledge distillation (DFKD). FedRevive integrates parameter-space aggregation with a server-side DFKD process that transfers knowledge from stale client updates to the current global model without access to data. A meta-learned generator synthesizes pseudo-samples for multi-teacher distillation. A hybrid aggregation scheme combining raw with DFKD updates effectively mitigates staleness while retaining AFL scalability. Experiments on vision and text benchmarks show that FedRevive achieves faster training by up to 29.7% and higher final accuracy by up to 11.4 percentage points than baselines.

## Repository Structure

```
.
├── README.md
├── data/                          # CIFAR-10/100 and the synthetic text snippets are stored here
└── codes/                         # run all commands from this directory
    ├── main_fedrevive.py          # entry point for FedRevive
    ├── main.py                    # entry point for all baselines and DD-FedRevive
    ├── server_fedrevive.py        # FedRevive server (hybrid aggregation, KD-Revive, class-proportion proxies)
    ├── client_fedrevive.py
    ├── server.py                  # asynchronous/synchronous FL server for the baselines
    ├── client.py
    ├── config.py                  # default values and documentation of every config option
    ├── requirements.txt
    ├── SampleConfigFiles/         # configs for all experiments in the paper
    │   └── {cifar10,cifar100,femnist,news}/{algorithm}.json
    ├── ExperimentOuts/            # default output directory
    └── utils/
        ├── dfkd_utils.py          # image KD-Revive: generator, synthesis, multi-teacher KD, β schedules
        ├── text_dfkd_utils.py     # text KD-Revive: soft-prompt synthesis for T5
        ├── synthetic_text_data_utils.py  # generates the text snippets used to initialize text synthesis
        ├── kd_utils.py            # DD-FedRevive (KD on real auxiliary data)
        ├── event_utils.py         # event queue for the simulated wall clock
        ├── data_utils.py          # datasets and non-IID partitioning
        ├── model_utils.py         # ResNet-18, FEMNIST CNN, T5-small classifier, server optimizers
        ├── exp_state_utils*.py    # checkpointing and resuming
        └── logging_utils.py
```

## Installation

A CUDA GPU is required for FedRevive.

Clone this repository, then from its root:

```bash
cd codes
conda create -n fedrevive python=3.13 -y
conda activate fedrevive
pip install -r requirements.txt
```

## Data

All datasets are downloaded automatically on first use; no manual preparation is needed.

| Code name | Dataset | Model | Source |
|---|---|---|---|
| `cifar10` | CIFAR-10 | ResNet-18 | `torchvision`, saved to `../data/cifar10` |
| `cifar100` | CIFAR-100 | ResNet-18 | `torchvision`, saved to `../data/cifar100` |
| `femnist` | FEMNIST (1000 writers) | CNN | Hugging Face `flwrlabs/femnist` via `flwr-datasets` |
| `news` | 20NewsGroups | T5-small + linear head | `sklearn.datasets.fetch_20newsgroups`; T5-small from Hugging Face |

FEMNIST, 20NewsGroups, and T5-small use the default Hugging Face and scikit-learn caches. Paths starting with `../data` are relative to `codes/`, so always launch runs from `codes/`.

**Text snippets for 20NewsGroups.** For text, FedRevive initializes the synthetic latent prompts from embeddings of short, high-temperature snippets generated by a small LM (Section 3.2 of the paper). The first FedRevive run on `news` generates these snippets automatically with `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (2,000 snippets for each of the 20 topics) and saves them to `../data/synthetic_news/synthetic_news.jsonl`. This step downloads the model from Hugging Face and needs a GPU. Later runs reuse the file. To generate the file once, before launching several seeds in parallel, run this from `codes/`:

```bash
python -m utils.synthetic_text_data_utils
```

## Running Experiments

Every experiment in the paper has a config file in `SampleConfigFiles/{dataset}/{algorithm}.json`. Run all commands from `codes/`.

**FedRevive** uses its own entry point:

```bash
python main_fedrevive.py --config SampleConfigFiles/{dataset}/fedrevive.json
```

**All other methods** (baselines and the DD-FedRevive ablation):

```bash
python main.py --config SampleConfigFiles/{dataset}/{algorithm}.json
```

For example:

```bash
# FedRevive on CIFAR-10
python main_fedrevive.py --config SampleConfigFiles/cifar10/fedrevive.json

# FedBuff on FEMNIST
python main.py --config SampleConfigFiles/femnist/fedbuff.json
```

### Algorithms

| Code name | Method | Entry point |
|---|---|---|
| `fedrevive` | **FedRevive (ours)** | `main_fedrevive.py` |
| `dd_fedrevive` | DD-FedRevive (data-driven ablation) | `main.py` |
| `afl_dw` | AFL-DW (staleness down-weighting ablation) | `main.py` |
| `fedavg` | FedAvg (synchronous) | `main.py` |
| `fedbuff` | FedBuff | `main.py` |
| `fadas` | FADAS | `main.py` |
| `fedac` | FedAC | `main.py` |
| `fedfa` | FedFa | `main.py` |

### Outputs

Each run writes to the `out_dir` set in its config (default `ExperimentOuts/{dataset}/{algorithm}`):

- `out.log`: training log
- `config.json`: the full resolved configuration
- `results.json`: test (and train) metrics against server round and simulated time
- `test_curves.png`: test accuracy and loss curves
- `ckpt.pth`, `checkpoint_exp_state/`: model checkpoint (every `save_model_interval` rounds) and full experiment state for resuming

Time in all logs and plots is simulated wall-clock time (see Appendix D of the paper), not real runtime.

### Seeds and resuming

- `setup_random_seed` controls model initialization, data partitioning, and client delay profiles; `run_random_seed` controls training randomness. The paper reports the mean over three seeds.
- The sample configs set `continue_exp_from` to their own `out_dir`, so restarting a preempted job resumes from the last saved state (written every `save_exp_state_interval` minutes). **When a run resumes, it loads the saved `config.json` from that directory and ignores the config file you passed.** To start a new run, such as another seed or different hyperparameters, change `out_dir` and `continue_exp_from` together, or set `continue_exp_from` to `null`. You can also resume any experiment directory with `--resume_from <dir>`.
- Runs stop when any of `max_rounds`, `max_wait_time` (simulated time), or `target_accuracy` is reached.
- Weights & Biases logging is off by default; set `wandb_flag: true` (and `wandb_entity`) to enable it.

### Key hyperparameters

Hyperparameters for every method and dataset are listed in Appendix J of the paper and are already set in the sample configs.

| Paper | Config key | Default in configs |
|---|---|---|
| $\eta_c$, $\eta_s$ | `local_learning_rate`, `server_lr` (top level) | per dataset (Table 6) |
| $\beta(\tau)$ | `beta` (`"cosine"` = 1 − cosine schedule) | `"cosine"` |
| $c$ (teacher buffer size) | `kd_buffer_size` | 8 |
| $T_{\text{gen}}$ | `generate_freq` | 10 |
| $K_{\text{synth}}$ | `g_steps` | 2 |
| $K_{\text{KD}}$ | `kd_nb_of_iters` | 10 |
| Synthesis / KD batch size | `synth_batch_size` / `kd_batch_size` | 64 / 32 |
| KD temperature $T$ | `T` | 1 |
| KD / generator / latent learning rates | `lr` / `lr_g` / `lr_z` | per dataset (Table 8) |
| $\alpha_{\text{target}}$, $\alpha_{\text{feature}}$, $\alpha_{\text{adv}}$ | `oh`, `bn`, `adv` | 1.0, per dataset, 0.1 |
| $T_{\text{probe}}$ (proxy temperature) | `estimation_temperature` (top level) | per dataset (Appendix B) |
| $T_{\text{est}}$ (uploads used for the proxy) | `how_many_dist_est_update_later` (top level) | 2 |

Keys not marked top level are inside `dfkd_settings`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{askin2026reviving,
  title     = {Reviving Stale Updates: Data-Free Knowledge Distillation for Asynchronous Federated Learning},
  author    = {Askin, Baris and Roth, Holger R. and Sun, Zhenyu and Joe-Wong, Carlee and Joshi, Gauri and Xu, Ziyue},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026}
}
```

## Contact

For questions, please open an issue or contact [Baris Askin](https://askinb.github.io).
