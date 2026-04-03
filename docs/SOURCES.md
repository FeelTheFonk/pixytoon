# Sources & Citations

## Diffusion & Acceleration

**Stable Diffusion** — Latent diffusion architecture enabling high-resolution image synthesis from text via a compressed latent space.
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

**LDM (Latent Diffusion Models)** — Foundation framework operating diffusion in a learned latent space rather than pixel space, reducing compute by orders of magnitude.
Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. [GitHub](https://github.com/CompVis/latent-diffusion)

**Hyper-SD** — Trajectory segmented consistency distillation reducing inference from 8 to 1-4 steps.
Ren, Y., et al. (2024). *Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis*. NeurIPS 2024. [arXiv:2404.13686](https://arxiv.org/abs/2404.13686)

**AnimateDiff** — Plug-and-play temporal motion module that adds video generation to any SD1.5 checkpoint.
Guo, Y., et al. (2023). *AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning*. ICLR 2024. [arXiv:2307.04725](https://arxiv.org/abs/2307.04725)

**AnimateDiff-Lightning** — Progressive adversarial distillation for 10x faster animation generation.
Lin, S., Liu, B., Li, J., & Yang, X. (2024). *AnimateDiff-Lightning: Cross-Model Diffusion Distillation*. [arXiv:2403.12706](https://arxiv.org/abs/2403.12706)

**FreeU** — Training-free skip connection re-weighting that improves generation quality at zero computational cost.
Si, C., Huang, Z., Jiang, Y., & Liu, Z. (2023). *FreeU: Free Lunch in Diffusion U-Net*. CVPR 2024. [arXiv:2309.11497](https://arxiv.org/abs/2309.11497)

**DeepCache** — Training-free feature caching across denoising steps yielding ~2x inference speedup.
Ma, X., Fang, G., & Wang, X. (2023). *DeepCache: Accelerating Diffusion Models for Free*. CVPR 2024. [arXiv:2312.00858](https://arxiv.org/abs/2312.00858)

**EquiVDM** — Equivariant variance-preserving noise schedule providing uniform SNR decay for improved diffusion training.
Chen, E., et al. (2024). *EquiVDM: Equivariant Diffusion for Molecule Generation in 3D*. [GitHub](https://github.com/eqvdm/eqvdm)

## Attention & Compilation

**FlashAttention-2** — IO-aware exact attention algorithm with optimized parallelism and work partitioning, eliminating quadratic memory overhead.
Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

**SageAttention2** — Efficient attention via INT4 quantization with outlier smoothing, further reducing attention compute on supported hardware.
Zhang, J., et al. (2025). *SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-Thread INT4 Quantization*. ICML 2025. [arXiv:2411.10958](https://arxiv.org/abs/2411.10958)

**torch.compile / Triton** — Graph-level JIT compilation generating fused Triton GPU kernels from PyTorch code.
Tillet, P., Kung, H. T., & Cox, D. (2019). *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*. MAPL 2019. [GitHub](https://github.com/triton-lang/triton)

**PAG (Perturbed-Attention Guidance)** — Self-rectifying sampling that perturbs intermediate attention maps to improve generation quality without classifier-free guidance overhead.
Ahn, D., et al. (2024). *Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance*. ECCV 2024. [arXiv:2403.17377](https://arxiv.org/abs/2403.17377)

## Image Processing

**ControlNet** — Auxiliary conditioning networks injecting spatial control (depth, edges, pose) into pretrained diffusion models.
Zhang, L., Rao, A., & Agrawala, M. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models*. ICCV 2023. [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)

**IP-Adapter** — Image prompt adapter enabling visual conditioning alongside text prompts in diffusion pipelines.
Ye, H., et al. (2023). *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models*. [arXiv:2308.06721](https://arxiv.org/abs/2308.06721)

**Real-ESRGAN** — Practical blind super-resolution trained on pure synthetic degradation data for real-world image upscaling.
Wang, X., et al. (2021). *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*. ICCV 2021 Workshop. [arXiv:2107.10833](https://arxiv.org/abs/2107.10833)

**RIFE** — Real-time intermediate flow estimation for high-quality video frame interpolation.
Huang, Z., et al. (2022). *Real-Time Intermediate Flow Estimation for Video Frame Interpolation*. ECCV 2022. [arXiv:2011.06294](https://arxiv.org/abs/2011.06294)

**PixelOE** — Pixel-art-aware outline extraction and edge detection for stylized downscaling.
[GitHub](https://github.com/KohakuBlueleaf/PixelOE)

**rembg / U2-Net** — Background removal via salient object detection with nested U-structures.
Qin, X., et al. (2020). *U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection*. Pattern Recognition, 106. [arXiv:2005.09007](https://arxiv.org/abs/2005.09007)

## Color & Post-Processing

**OKLAB** — Perceptually uniform color space for accurate color distance computation, replacing CIELAB in all color pipelines.
Ottosson, B. (2020). *A perceptual color space for image processing*. [Blog](https://bottosson.github.io/posts/oklab/)

**Floyd-Steinberg Dithering** — Error-diffusion dithering algorithm distributing quantization error to neighboring pixels.
Floyd, R. W. & Steinberg, L. (1976). *An Adaptive Algorithm for Spatial Greyscale*. Proceedings of the Society for Information Display, 17(2), 75-77.

**Bayer Dithering** — Ordered dithering using threshold matrices for deterministic, parallelizable quantization.
Bayer, B. E. (1973). *An Optimum Method for Two-Level Rendition of Continuous-Tone Pictures*. IEEE ICC 1973.

**K-Means Clustering** — Iterative centroid-based clustering used for optimal color palette extraction and quantization.
Lloyd, S. P. (1982). *Least Squares Quantization in PCM*. IEEE Transactions on Information Theory, 28(2), 129-137.

## Audio & DSP

**librosa** — Core audio analysis library providing spectral features, onset detection, and beat tracking.
McFee, B., et al. (2015). *librosa: Audio and Music Signal Analysis in Python*. SciPy 2015. [GitHub](https://github.com/librosa/librosa)

**Demucs / HTDemucs** — Hybrid transformer architecture for music source separation (vocals, drums, bass, other).
Rouard, S., Massa, F., & Defossez, A. (2023). *Hybrid Transformers for Music Source Separation*. ICASSP 2023. [arXiv:2211.08553](https://arxiv.org/abs/2211.08553)

**ITU-R BS.1770** — International standard for loudness measurement and true-peak detection using K-weighted filtering.
ITU (2015). *Algorithms to measure audio programme loudness and true-peak audio level*. ITU-R BS.1770-4. [Standard](https://www.itu.int/rec/R-REC-BS.1770)

**K-Weighting** — Two-stage pre-emphasis filter (head-related shelf + high-pass) defined in BS.1770 for perceptual loudness measurement.
ITU (2015). *ITU-R BS.1770-4*, Annex 1. [Standard](https://www.itu.int/rec/R-REC-BS.1770)

## Infrastructure

**Diffusers** — Hugging Face library providing pretrained diffusion pipelines, schedulers, and model loading.
von Platen, P., et al. (2022). *Diffusers: State-of-the-art diffusion models*. [GitHub](https://github.com/huggingface/diffusers)

**PyTorch** — Tensor computation and automatic differentiation framework underlying all inference.
Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 2019. [pytorch.org](https://pytorch.org)

**Numba** — LLVM-based JIT compiler used for vectorized audio analysis and post-processing hot loops.
Lam, S. K., Pitrou, A., & Seibert, S. (2015). *Numba: A LLVM-based Python JIT Compiler*. LLVM-HPC 2015. [numba.pydata.org](https://numba.pydata.org)

**simpleeval** — Sandboxed single-expression evaluator for user-defined parameter expressions.
Sherrill, D. (2013). *simpleeval*. [GitHub](https://github.com/danthedeckie/simpleeval)

**OpenCV** — Computer vision primitives used for motion warping, perspective transforms, and frame manipulation.
Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal. [opencv.org](https://opencv.org)
