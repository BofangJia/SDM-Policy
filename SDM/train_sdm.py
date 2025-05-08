if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import itertools
import os

import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import torch.nn.functional as F
from typing import Generator
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D


OmegaConf.register_new_resolver("eval", eval, replace=True)


# Compute score function
def score_function(x_t, unet_output, alpha_t, sigma_t):
   
    alpha_t = alpha_t.reshape(-1, 1, 1)
    sigma_t = sigma_t.reshape(-1, 1, 1)

    return -(x_t - alpha_t * unet_output) / (sigma_t ** (2))


class Solver:
    def __init__(self, alpha_cumprods: np.ndarray, timesteps: int = 1000, ddim_timesteps: int = 50) -> None:
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device: torch.device) -> "Solver":
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self
    

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer_generator = hydra.utils.instantiate(
            cfg.optimizer_generator, params=self.model.parameters())
        
        self.optimizer_dynamic_unet = hydra.utils.instantiate(
            cfg.optimizer_dynamic_unet, params=self.model.parameters())

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        
        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
        else:
            raise ValueError(f"Training Must Have A Teacher Model !!!!!")

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

    
        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)

        noise_scheduler = self.model.noise_scheduler
        alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
        sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)


        solver = Solver(
            noise_scheduler.alphas_cumprod.numpy(),
            timesteps=noise_scheduler.config.num_train_timesteps,
            ddim_timesteps=cfg.policy.num_inference_steps,
        )

        
        encoder = self.model.obs_encoder
        encoder.requires_grad_(False)

        unet = self.model.model
        generator = ConditionalUnet1D(*self.model.unet_config)
        generator.load_state_dict(unet.state_dict())
        generator.requires_grad_(True)


        dynamic_unet = ConditionalUnet1D(*self.model.unet_config)
        dynamic_unet.load_state_dict(unet.state_dict())
        dynamic_unet.requires_grad_(True)
        pre_trained_unet = ConditionalUnet1D(*self.model.unet_config)
        pre_trained_unet.load_state_dict(unet.state_dict())
        pre_trained_unet.requires_grad_(False)

        generator = generator.to(device)
        pre_trained_unet = pre_trained_unet.to(device)
        dynamic_unet = dynamic_unet.to(device)

        self.model.model = generator
        
        normalizer = self.model.normalizer
        normalizer.requires_grad_(False)

        alpha_schedule = alpha_schedule.to(device)
        sigma_schedule = sigma_schedule.to(device)
        
        solver = solver.to(device)
        
        generator_optimizer = torch.optim.AdamW(
            generator.parameters(),
            lr=cfg.optimizer_generator.lr,
            betas=(cfg.optimizer_generator.betas[0], cfg.optimizer_generator.betas[1]),
            weight_decay=cfg.optimizer_generator.weight_decay,
            eps=cfg.optimizer_generator.eps
        )

        dynamic_unet_optimizer = torch.optim.AdamW(
            dynamic_unet.parameters(),
            lr=cfg.optimizer_dynamic_unet.lr,
            betas=(cfg.optimizer_dynamic_unet.betas[0], cfg.optimizer_dynamic_unet.betas[1]),
            weight_decay=cfg.optimizer_dynamic_unet.weight_decay,
            eps=cfg.optimizer_dynamic_unet.eps
        )


        lr_generator_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=generator_optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every
        )


        lr_dynamic_unet_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=dynamic_unet_optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every
        )

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
    
        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # save batch for sampling
        train_sampling_batch = None

        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        self.global_step = 0
        self.epoch = 0
             
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            train_losses = list()
            diffusion_losses = list()
            # ========= train for this epoch ==========
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                    # normalize input
                    nobs = normalizer.normalize(batch['obs'])
                    nactions = normalizer['action'].normalize(batch['action'])

                    if not self.model.use_pc_color:
                        nobs['point_cloud'] = nobs['point_cloud'][..., :3]
                    
                    batch_size = nactions.shape[0]
                    horizon = nactions.shape[1]

                    # handle different ways of passing observation
                    local_cond = None
                    global_cond = None
                    trajectory = nactions
                    cond_data = trajectory
                    
                    if self.model.obs_as_global_cond:
                        # reshape B, T, ... to B*T
                        # [batch_size * n_bos_steps, 512, 3], [batch_size * n_bos_steps, 24]
                        this_nobs = dict_apply(nobs, 
                            lambda x: x[:,:self.model.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                        nobs_features = encoder(this_nobs)

                        if "cross_attention" in self.model.condition_type:
                            # treat as a sequence
                            global_cond = nobs_features.reshape(batch_size, self.model.n_obs_steps, -1)
                        else:
                            # reshape back to B, Do
                            global_cond = nobs_features.reshape(batch_size, -1)

                        this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
                        this_n_point_cloud = this_n_point_cloud[..., :3]

                        trajectory = cond_data.detach()

                    noise = torch.randn(trajectory.shape, device=trajectory.device) 
    
                    generator_output = generator(
                        sample=noise, 
                        timestep=100, 
                        local_cond=local_cond, 
                        global_cond=global_cond)
                    
                    index = torch.randint(0, cfg.policy.num_inference_steps, (batch_size,), device=device).long()
                    timesteps = solver.ddim_timesteps[index] 
                    timesteps = timesteps.to(device)  
                    noise = torch.randn(generator_output.shape, device=generator_output.device)
                    score_function_input = noise_scheduler.add_noise(generator_output, noise, timesteps)

                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))


                    with torch.no_grad():
                        pre_trained_output = pre_trained_unet(
                            sample=score_function_input, 
                            timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


                    dynamic_output = dynamic_unet(
                        sample=score_function_input, 
                        timestep=timesteps, 
                        local_cond=local_cond, 
                        global_cond=global_cond)
                    

                    alpha = alpha_schedule[timesteps]
                    sigma = sigma_schedule[timesteps]
                    # Calculate the average of alpha and sigma
                    alpha= alpha.to(device)
                    sigma= sigma.to(device)

                    # Calculate the output of pre_trained_unet and dynamic_unet
                    with torch.no_grad():
                        s_pre_trained = score_function(score_function_input, pre_trained_output, alpha, sigma)

                    s_dynamic = score_function(score_function_input, dynamic_output, alpha, sigma)

                    grad = (s_dynamic - s_pre_trained) / torch.abs(s_pre_trained).mean()
                    grad = torch.nan_to_num(grad)


                    # Update the fake_unet every 5 epochs
                    if self.epoch % 5 == 0:

                        # else:  
                        loss = 0.5 * F.mse_loss(generator_output.float(), (generator_output-grad).detach().float(), reduction="mean") 
                        loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.training.max_grad_norm)
                        generator_optimizer.step()
                        generator_optimizer.zero_grad()


                    lr_generator_scheduler.step()


                    # Always backpropagate the fake_loss
                    diffusion_loss = torch.mean((dynamic_output.float() - generator_output.float())**2) 
                    diffusion_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(dynamic_unet.parameters(), cfg.training.max_grad_norm)
                    dynamic_unet_optimizer.step()
                    dynamic_unet_optimizer.zero_grad()
                    lr_dynamic_unet_scheduler.step()
 


                    raw_loss_cpu = loss.item()
                    diffusion_loss_cpu = diffusion_loss.item()
                    tepoch.set_postfix(generator_loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    tepoch.set_postfix(diffusion_loss=diffusion_loss_cpu, refresh=False)
                    diffusion_losses.append(diffusion_loss_cpu)

                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'diffusion_losses':diffusion_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr_generator': lr_generator_scheduler.get_last_lr()[0],
                        'lr_dynamic_unet': lr_dynamic_unet_scheduler.get_last_lr()[0]
                    }
                    generator_loss_dict = {'bc_loss': loss.item()}
                    step_log.update(generator_loss_dict)



            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            
            # ========= eval for this epoch ==========
            policy = self.model
            policy.eval()

            # run rollout
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                t3 = time.time()
                # runner_log = env_runner.run(policy, dataset=dataset)
                runner_log = env_runner.run(policy, use_consistency_model=self.cfg.use_consistency_model)
                t4 = time.time()
                # print(f"rollout time: {t4-t3:.3f}")
                # log all
                step_log.update(runner_log)
                
            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            
            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    result = policy.predict_action(obs_dict, True)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if env_runner is None:
                step_log['test_mean_score'] = - train_loss
                
            policy.train()
            

            # end of epoch
            # log of last step is combined with validation and rollout
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)

        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path, tag="latest")

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy, use_consistency_model=self.cfg.use_consistency_model)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='newest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest' or tag=='newest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
                      

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='newest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
