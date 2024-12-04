# [Score and Distribution Matching Policy: Advanced accelerated Visuomotor Policies via matched distillation](https://bofangjia1227.github.io/page/)


[Project Page](https://bofangjia1227.github.io/page/) | [arXiv](https://bofangjia1227.github.io/page/) | [Paper](https://bofangjia1227.github.io/page/)

[Bofang Jia](https://bofangjia1227.github.io/page/)\*, [Can Cui](https://bofangjia1227.github.io/page/)\*, [Pengxiang Ding](https://bofangjia1227.github.io/page/)\*, [Mingyang Sun](https://bofangjia1227.github.io/page/), [Pengfang Qian](https://bofangjia1227.github.io/page/), [Siteng Huang](https://bofangjia1227.github.io/page/), [Zhaoxin Fan](https://bofangjia1227.github.io/page/), [Donglin Wang](https://bofangjia1227.github.io/page/)<sup>†</sup>


![](./files/sdm.svg) 

<b>Overview of SDM Policy</b>: Our method distills the Diffusion Policy, which requires long inference times and high computational costs, into a fast and stable one-step generator. Our SDM Policy is represented by the one-step generator, which requires continual correction and optimization via the Corrector during training, but relies solely on the generator during evaluation. The corrector's optimization is based on two components: gradient optimization and diffusion optimization. The gradient optimization part primarily involves optimizing the entire distribution by minimizing the KL divergence between two distributions, $P_{\theta}$ and $D_{\theta}$, with distribution details represented through a score function that guides the gradient update direction, providing a clear signal. The diffusion optimization component enables $D_{\theta}$ to quickly track changes in the one-step generator’s output, maintaining consistency. Details on loading observational data for both evaluation and training processes are provided above the diagram. Our method applies to both 2D and 3D scenarios.

# 💻 Installation

See [INSTALL.md](INSTALL.md) for installation instructions. 

# 🛠️ Usage




# 📚 Checkpoints


# 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

# 😺 Acknowledgement
Our code is generally built upon: [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [DexMV](https://github.com/yzqin/dexmv-sim), [DexArt](https://github.com/Kami-code/dexart-release), [VRL3](https://github.com/microsoft/VRL3), [DAPG](https://github.com/aravindr93/hand_dapg), [DexDeform](https://github.com/sizhe-li/DexDeform), [RL3D](https://github.com/YanjieZe/rl3d), [GNFactor](https://github.com/YanjieZe/GNFactor), [H-InDex](https://github.com/YanjieZe/H-InDex), [MetaWorld](https://github.com/Farama-Foundation/Metaworld), [BEE](https://jity16.github.io/BEE/), [Bi-DexHands](https://github.com/PKU-MARL/DexterousHands), [HORA](https://github.com/HaozhiQi/hora). We thank all these authors for their nicely open sourced code and their great contributions to the community.

Contact [Bofang Jia](https://bofangjia1227.github.io/page/) if you have any questions or suggestions.

# 📝 Citation

If you find our work useful, please consider citing:




