# Overview
This is a 585 Project directory which includes scripts/files to be run on Greatlakes/CloudLab. Each directory normally contains a self contained experiment, which can be run with the sbat files for Greatlakes.

# Summary of Important Code
## Experiments
### greatlakes/residual_stream
Code for prepping a small set of 1000 random GPT2 activations to use in testing SAE training.
### greatlakes/basic_encoder_decoder_timing
Contains initial testing with using sparse matrix formats and timing the encoder/decoder time for SAEs.
### greatlakes/cusparse_profiling
More extensive testing with varying sparsity and batch size for the sparse and dense decoder types.
### greatlakes/bspmm_kernel
Code for the batched sparse matrix multiplication kernel, along with testing correctness and decoding time for the SAEs.
#### Batched SpMM Triton Kernel
See [this file](https://github.com/Nathan-Yap/CSE585_sae_project/blob/main/greatlakes/bspmm_kernel/sparse_triton_kernel.py) for the kernel implementation, which is used by the surrounding files for profiling.
### Other folders
Contain tests that didn't end up with useful results (e.x. CloudLab tests and older tests with different profiling tools).
