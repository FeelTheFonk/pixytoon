# Sources

Scientific papers, technical references, and key dependencies behind SDDj.

---

## Diffusion & Acceleration

**Stable Diffusion**
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022.
[arXiv:2112.10752](https://arxiv.org/abs/2112.10752) · [GitHub](https://github.com/CompVis/latent-diffusion)

**Classifier-Free Guidance**
Ho, J. & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. NeurIPS 2021 Workshop.
[arXiv:2207.12598](https://arxiv.org/abs/2207.12598)

**Hyper-SD** — Trajectory segmented consistency distillation (8 → 1–4 steps)
Ren, Y., Xia, X., Lu, Y., Zhang, J., Wu, J., Xie, P., Wang, X., & Xiao, X. (2024). *Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis*. NeurIPS 2024.
[arXiv:2404.13686](https://arxiv.org/abs/2404.13686)

**AnimateDiff** — Temporal motion module for video generation
Guo, Y., Yang, C., Rao, A., Wang, Y., Qiao, Y., Lin, D., & Dai, B. (2023). *AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning*. ICLR 2024.
[arXiv:2307.04725](https://arxiv.org/abs/2307.04725) · [GitHub](https://github.com/guoyww/AnimateDiff)

**AnimateDiff-Lightning** — Progressive adversarial distillation (10× faster animation)
Lin, S., Liu, B., Li, J., & Yang, X. (2024). *AnimateDiff-Lightning: Cross-Model Diffusion Distillation*.
[arXiv:2403.12706](https://arxiv.org/abs/2403.12706) · [HuggingFace](https://huggingface.co/ByteDance/AnimateDiff-Lightning)

**DeepCache** — Feature caching across denoising steps (~2× speedup, training-free)
Ma, X., Fang, G., & Wang, X. (2023). *DeepCache: Accelerating Diffusion Models for Free*. CVPR 2024.
[arXiv:2312.00858](https://arxiv.org/abs/2312.00858) · [GitHub](https://github.com/horseee/DeepCache)

**FreeU** — Skip connection re-weighting for quality enhancement (zero cost)
Si, C., Huang, Z., Jiang, Y., & Liu, Z. (2023). *FreeU: Free Lunch in Diffusion U-Net*. CVPR 2024.
[arXiv:2309.11497](https://arxiv.org/abs/2309.11497) · [GitHub](https://github.com/ChenyangSi/FreeU)

**FreeInit** — Temporal initialization improvement for video diffusion
Wu, T., Si, C., Jiang, Y., Huang, Z., & Liu, Z. (2023). *FreeInit: Bridging Initialization Gap in Video Diffusion Models*. CVPR 2024.
[arXiv:2312.07537](https://arxiv.org/abs/2312.07537)

**ControlNet** — Spatial conditioning via auxiliary networks
Zhang, L., Rao, A., & Agrawala, M. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models*. ICCV 2023.
[arXiv:2302.05543](https://arxiv.org/abs/2302.05543) · [GitHub](https://github.com/lllyasviel/ControlNet)

**LoRA** — Low-rank adaptation for efficient fine-tuning
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

---

## Attention & Compilation

**FlashAttention** — IO-aware exact attention algorithm
Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

**FlashAttention-2** — Improved parallelism and work partitioning
Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*.
[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

**Triton** — Language and compiler for parallel programming
Tillet, P., Kung, H. T., & Cox, D. (2019). *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*. MAPL 2019.
[GitHub](https://github.com/triton-lang/triton)

**TF32 Precision** — Tensor Float 32 for Ampere+ GPUs
NVIDIA (2020). *NVIDIA A100 Tensor Core GPU Architecture*.
[Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)

---

## Audio & DSP

**librosa** — Audio analysis library
McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). *librosa: Audio and Music Signal Analysis in Python*. SciPy 2015.
[DOI:10.25080/Majora-7b98e3ed-003](https://doi.org/10.25080/Majora-7b98e3ed-003) · [GitHub](https://github.com/librosa/librosa)

**Demucs / HTDemucs** — Hybrid transformer music source separation
Défossez, A. (2021). *Hybrid Spectrogram and Waveform Source Separation*. ISMIR 2021 MDX Workshop.
Rouard, S., Massa, F., & Défossez, A. (2023). *Hybrid Transformers for Music Source Separation*. ICASSP 2023.
[arXiv:2211.08553](https://arxiv.org/abs/2211.08553) · [GitHub](https://github.com/facebookresearch/demucs)

**SuperFlux Onset Detection** — Vibrato-robust onset detection
Böck, S. & Widmer, G. (2013). *Maximum Filter Vibrato Suppression for Onset Detection*. DAFx-13.
[Paper](http://dafx13.nuim.ie/papers/09.dafx2013_submission_12.pdf)

**ITU-R BS.1770** — Loudness measurement and K-weighting
ITU (2015). *Algorithms to measure audio programme loudness and true-peak audio level*. ITU-R BS.1770-4.
[Standard](https://www.itu.int/rec/R-REC-BS.1770)

**Savitzky-Golay Filter** — Polynomial smoothing
Savitzky, A. & Golay, M. J. E. (1964). *Smoothing and Differentiation of Data by Simplified Least Squares Procedures*. Analytical Chemistry, 36(8), 1627–1639.
[DOI:10.1021/ac60214a047](https://doi.org/10.1021/ac60214a047)

**CQT Chromagram** — Constant-Q transform for pitch class analysis
Schörkhuber, C. & Klapuri, A. (2010). *Constant-Q Transform Toolbox for Music Processing*. SMC 2010.

**pyloudnorm** — LUFS loudness normalization
Steinmetz, C. J. (2019). *pyloudnorm: A simple implementation of the ITU-R BS.1770-4 loudness algorithm in Python*.
[GitHub](https://github.com/csteinmetz1/pyloudnorm)

**madmom** — RNN-based beat tracking (optional backend)
Böck, S., Korzeniowski, F., Schlüter, J., Krebs, F., & Widmer, G. (2016). *madmom: a New Python Audio and Music Signal Processing Library*. ACM MM 2016.
[GitHub](https://github.com/CPJKU/madmom)

---

## Post-Processing

**U²-Net** — Salient object detection for background removal
Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M. (2020). *U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection*. Pattern Recognition, 106.
[arXiv:2005.09007](https://arxiv.org/abs/2005.09007) · [GitHub](https://github.com/xuebinqin/U-2-Net)

**Floyd-Steinberg Dithering** — Error-diffusion dithering
Floyd, R. W. & Steinberg, L. (1976). *An Adaptive Algorithm for Spatial Greyscale*. Proceedings of the Society for Information Display, 17(2), 75–77.

**CIELAB Color Space** — Perceptual color distance
CIE (1976). *Recommendations on Uniform Color Spaces, Color-Difference Equations, Psychometric Color Terms*. CIE Publication No. 15, Supplement 2.

**Bayer Dithering** — Ordered dithering matrices
Bayer, B. E. (1973). *An Optimum Method for Two-Level Rendition of Continuous-Tone Pictures*. IEEE ICC 1973.

**K-Means Clustering** — Color quantization via cluster analysis
Lloyd, S. P. (1982). *Least Squares Quantization in PCM*. IEEE Transactions on Information Theory, 28(2), 129–137.

---

## Infrastructure

**Diffusers** — Hugging Face diffusion model library
von Platen, P., Patil, S., Lozhkov, A., et al. (2022). *Diffusers: State-of-the-art diffusion models*.
[GitHub](https://github.com/huggingface/diffusers)

**PyTorch** — Deep learning framework
Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 2019.
[pytorch.org](https://pytorch.org)

**FastAPI** — Modern Python web framework
Ramírez, S. (2018). *FastAPI*.
[GitHub](https://github.com/tiangolo/fastapi)

**Numba** — JIT compilation for Python
Lam, S. K., Pitrou, A., & Seibert, S. (2015). *Numba: A LLVM-based Python JIT Compiler*. LLVM-HPC 2015.
[numba.pydata.org](https://numba.pydata.org)

**rembg** — Background removal wrapper
Gatis, D. (2020). *rembg: Remove image background*.
[GitHub](https://github.com/danielgatis/rembg)

**simpleeval** — Safe expression evaluation
Sherrill, D. (2013). *simpleeval: A simple, safe single expression evaluator library*.
[GitHub](https://github.com/danthedeckie/simpleeval)

**OpenCV** — Computer vision (motion warp, perspective tilt)
Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal.
[opencv.org](https://opencv.org)
