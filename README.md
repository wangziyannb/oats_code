# OATS Reproducible Code
This folder contains the large language model code utilized for the paper: OATS: Outlier-Aware Pruning through Sparse and Low Rank Decomposition.

# Requirements
The code was run using Python 3.10.12. The packages used to run the experiments can be found in the `requirements.txt` file, which contains all the dependencies for setting up the virtual environment.

```
pip install -r requirements.txt
```

# How to Run?
The configurations for each experiment can be found under the file `./config_files/prune_configs.py`. Each experiment can be run by calling
```
# 1 <= run_id <= 93
python run_prune.py run_id 
```
where `run_id` is a positive integer that indexes a specific experiment. Prior to running, in `run_prune.py`, fill in where to save the results (`results_path`) and where to save the models for checkpointing during pruning (`checkpoint_path`). 

# Codebases Utilized
Our code utilizes and takes inspiration from the codebases found at the following GitHub Repos:
    SliceGPT: https://github.com/microsoft/TransformerCompression
    SparseGPT: https://github.com/IST-DASLab/sparsegpt
    Wanda: https://github.com/locuslab/wanda
    DSNoT: https://github.com/zyxxmu/DSnoT
    OWL: https://github.com/luuyin/OWL

# OATS dual-branch quantization

See [OATS_AWQ.md](OATS_AWQ.md) for post-hoc and in-loop AWQ-style fake quantization of the native OATS sparse and low-rank branches. The original OATS weighted SVD, pruning rules, update order, and last-iteration selection are retained. Entry point: `python run_oats_awq.py --help`.

See [OATS_REFINE.md](OATS_REFINE.md) for fixed-mask quantization refinement, which optimizes integer codes and group steps with optional cross-branch pair updates. It includes first-block rank 32/128 comparisons, full-model calibration and PPL evaluation, tests, and reproduction commands. Entry points: `python run_oats_refine.py --help` (projection probe) and `python run_oats_refine_full.py --help` (full model). Existing OATS/AWQ optimization paths are unchanged.
