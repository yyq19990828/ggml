# SAM downstream GGML adaptations

This fork retains GGML v0.26.0 at upstream commit
`d7cb574130e6f01ad25b3289685489200febcd74` and the downstream runtime changes
imported from SAM.cpp commit `c51cc27`. GGML's MIT license remains in the root
`LICENSE`. The following adapted portions have their own retained terms.

## PyTorch CUDA arithmetic

The persistent/register and generic F32 softmax implementations in
`sam-precise.cu`, `softmax-order.cu` and `softmax-wide-order.cu` adapt PyTorch
CUDA reduction arithmetic. Source references include PyTorch v2.10.0
`aten/src/ATen/native/cuda/SoftMax.cu` and
`aten/src/ATen/native/cuda/block_reduce.cuh`:

- https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/SoftMax.cu
- https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/block_reduce.cuh

The group-normalization arithmetic also follows the PyTorch-derived source
investigation documented by SAM.cpp. The complete retained upstream
copyright and BSD-style license text is in `LICENSE.pytorch.txt`. This fork
does not replace those terms with GGML's MIT license.

## Grounding DINO deformable attention

`ms-deform-attn.cu` adapts
`groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_im2col_cuda.cuh`
from Grounding DINO commit `856dde20aee659246248e20734ef9ba5214f5e44`:

https://github.com/IDEA-Research/GroundingDINO/tree/856dde20aee659246248e20734ef9ba5214f5e44

The source retains the original notices: Copyright (c) 2020 SenseTime and
Copyright (c) 2018 Microsoft for the DCN implementation. These adapted
portions follow Apache License 2.0, retained in
`LICENSE.grounding-dino.txt`. The source header's reference to its license
means these Apache terms, not the root GGML MIT license.

## Metal window operators

The downstream Metal window partition/restoration implementation is adapted
from the MIT-licensed PABannier GGML revision
`7a466633195d336e08dd5fec8b8ab42a2c5cb5e3`:

https://github.com/PABannier/ggml/tree/7a466633195d336e08dd5fec8b8ab42a2c5cb5e3

Its license is identical to the retained root GGML license, including
Copyright (c) 2023-2026 The ggml authors.

## Optional NVIDIA cuDNN

`GGML_CUDA_CUDNN=ON` requires an externally supplied NVIDIA cuDNN 9.10.2
SDK/runtime. Headers and binaries are not bundled, downloaded or
redistributed by this fork. Obtain and use that optional dependency under
NVIDIA's applicable terms, independently of the source licenses above.
