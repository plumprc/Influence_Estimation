# Which Influence Are We Estimating? The Role of Counterfactual Specifications in Data Attribution

## 🧩 Overview

This repository contains the code for studying how counterfactual specifications shape data influence estimation. Influence estimators are often compared as if they approximated the same quantity, but they can implicitly target different behaviors, interventions, and retraining processes. The code here implements controlled exact-influence experiments, noisy-label detection experiments, and ScienceQA attribution experiments with instruction-tuned LLMs.

## 🗂️ Repository Structure

| Directory | Description |
|---|---|
| `Controlled_exp/` | Controlled experiments with exact one-step, multi-step, and inverse-Hessian counterfactual updates on FashionMNIST and CIFAR-10 |
| `LLM_exp/` | ScienceQA response-corruption and conditional-backdoor attribution experiments with Qwen3-8B, Gemma-2-9B-it, and Llama-3.1-8B-Instruct |
| `NoisyLabel_exp/` | Noisy-label detection, architecture, and removal experiments on CIFAR-10 |

Each directory is self-contained and has its own README with reproduction instructions, experiment scopes, and output conventions.

## 🛠️ Setup

Dependencies are pinned to the versions used for the experiments. Install them with:

```bash
python -m pip install -r requirements.txt
```

The controlled and noisy-label experiments expect the standard `torchvision` dataset layout and can download FashionMNIST and CIFAR-10 automatically. The LLM experiments use [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) and instruction-tuned checkpoints from Hugging Face, which can be cached in advance with `hf download`.

## 🔬 Reproduction

All paper-facing results are produced through one shell entry point per experiment family.

```bash
# Controlled exact-influence experiments
cd Controlled_exp
bash shell/run_fmnist_exp.sh
bash shell/run_cifar10_exp.sh

# Noisy-label experiments
cd ../NoisyLabel_exp
bash shell/run_noisy_label_exp.sh

# ScienceQA LLM experiments
cd ../LLM_exp
MODEL=qwen3-8b bash shell/run_llm_exp1.sh
MODEL=qwen3-8b bash shell/run_llm_exp2.sh
```

The launchers skip completed seed-level outputs automatically, aggregate formal result bundles, and regenerate the figures. See the README in each directory for the full protocol, supported overrides, and optional analyses. LLM experiments require a CUDA GPU with sufficient memory for 8B-scale models.

## 📚 Citation

```
@article{li2026influence,
  title={Which Influence Are We Estimating? The Role of Counterfactual Specifications in Data Attribution},
  author={Li, Zhe and Zhao, Wei and Zhang, Peixin and Sun, Jun},
  journal={arXiv preprint arXiv:2609.31214},
  year={2026}
}
```
