# [Score and Distribution Matching Policy: Advanced Accelerated Visuomotor Policies via Matched Distillation](https://bofangjia1227.github.io/page/)


[Project Page](https://bofangjia1227.github.io/page/) | [arXiv](https://bofangjia1227.github.io/page/) | [Paper](https://bofangjia1227.github.io/page/) | [Data](https://bofangjia1227.github.io/page/)

[Bofang Jia](https://bofangjia1227.github.io/page/)\*, [Pengxiang Ding](https://dingpx.github.io/)\*, [Can Cui](http://cuixxx.github.io/)\*,  [Mingyang Sun](https://bofangjia1227.github.io/page/), [Pengfang Qian](https://bofangjia1227.github.io/page/), [Siteng Huang](https://kyonhuang.top/), [Zhaoxin Fan](https://zhaoxinf.github.io/), [Donglin Wang](https://milab.westlake.edu.cn/index.html)<sup>†</sup>


![](./files/sdm.svg) 

<b>Overview of SDM Policy</b>: Our method distills the Diffusion Policy, which requires long inference times and high computational costs, into a fast and stable one-step generator. Our SDM Policy is represented by the one-step generator, which requires continual correction and optimization via the Corrector during training, but relies solely on the generator during evaluation. The corrector's optimization is based on two components: gradient optimization and diffusion optimization. The gradient optimization part primarily involves optimizing the entire distribution by minimizing the KL divergence between two distributions, $P_{\theta}$ and $D_{\theta}$, with distribution details represented through a score function that guides the gradient update direction, providing a clear signal. The diffusion optimization component enables $D_{\theta}$ to quickly track changes in the one-step generator’s output, maintaining consistency. Details on loading observational data for both evaluation and training processes are provided above the diagram. Our method applies to both 2D and 3D scenarios.

# 💻 Installation

See [INSTALL.md](INSTALL.md) for installation instructions. 

# 🛠️ Usage

Scripts for generating demonstrations, training, and evaluation are all provided in the `scripts/` folder. 

The results are logged by `wandb`, so you need to `wandb login` first to see the results and videos.

For more detailed parameter Settings and descriptions, refer to scripts and codes. We have provided a simple illustration of using the code base here.


1. Generate demonstrations by `gen_demonstration_adroit.sh` and `gen_demonstration_dexart.sh`. See the scripts for details. For example:
    ```bash
    bash scripts/gen_demonstration_adroit.sh hammer
    ```
    This will generate demonstrations for the `hammer` task in Adroit environment. The data will be saved in `SDM/data/` folder automatically.


2. Train and evaluate a teacher policy with behavior cloning. For example:
    ```bash
    # bash scripts/train_policy.sh config_name task_name addition_info seed gpu_id 
    bash scripts/train_policy.sh dp3 adroit_hammer 0901 0 0
    ```
    This will train a DP3 policy on the `hammer` task in Adroit environment using point cloud modality. By default we **save** the ckpt (optional in the script). 
   
3. Copy teacher's ckpt. For example:
    ```bash
    # bash scripts/cp_teacher.sh alg_name task_name teacher_addition_info addition_info seed gpu_id
    bash scripts/cp_teacher.sh dp3_sdm adroit_hammer 0901 0901_sdm 0 0
    ```
    This will create a folder for student model training and copy the checkpoint for the specified teacher model here.
    
4. Train and evaluate SDM Policy. For example:
    ```bash
    # bash scripts/train_policy_sdm.sh config_name task_name addition_info seed gpu_id
    bash scripts/train_policy_sdm.sh dp3_sdm adroit_hammer 0901_sdm 0 0
    ```
    This will train SDM Policy use a DP3 policy teacher model on the `hammer` task in Adroit environment using point cloud modality. 





# 📚 Checkpoints

The checkpoints for all 57 tasks in the paper can be found here
[Adroit](https://drive.google.com/drive/folders/1Fq2PM9PqBWAEwPcOZdHZvrxaZtB6VC6W?usp=drive_link),
[DexArt](https://drive.google.com/drive/folders/1GrpyF3MD__nd6h_0tQE6-K-guJZBAD2P?usp=drive_link),
[MetaWorld](https://drive.google.com/drive/folders/1eVOfn__UEzFcPyO6pC_7y3BMYyctMSz9?usp=drive_link).

We are uploading gradually, sorry to say that due to the total checkpoints of about 600G, we are actively looking for a public storage platform for storage. Please contact us for priority updates if necessary.

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

# 🙏 Acknowledgement
Our code is generally built upon: [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), [ManiCM](https://github.com/ManiCM-fast/ManiCM), [DMD](https://github.com/tianweiy/DMD2), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [DexArt](https://github.com/Kami-code/dexart-release), [VRL3](https://github.com/microsoft/VRL3), [MetaWorld](https://github.com/Farama-Foundation/Metaworld). We thank all these authors for their nicely open sourced code and their great contributions to the community.

Contact [Bofang Jia](https://bofangjia1227.github.io/page/) if you have any questions or suggestions.

# 📝 Citation

If you find our work useful, please consider citing:
```
@misc{jia2024scoredistributionmatchingpolicy,
      title={Score and Distribution Matching Policy: Advanced Accelerated Visuomotor Policies via Matched Distillation}, 
      author={Bofang Jia and Pengxiang Ding and Can Cui and Mingyang Sun and Pengfang Qian and Zhaoxin Fan and Donglin Wang},
      year={2024},
      eprint={2412.09265},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2412.09265}, 
}

```



