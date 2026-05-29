import collections
import dill
import math
import pathlib
from collections import deque

import numpy as np
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

class PushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            perturb_level=0.0,
            perturb_type=0
        ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size,
                        perturb_level=perturb_level,
                        perturb_type=perturb_type
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseImagePolicy, extract_dynamic_features=None, action_latent_encoder=None, action_latent_decoder=None, history=5):
        # DCDP inference needs three extra modules in addition to the diffusion policy:
        # a visual dynamics extractor, a VAE encoder for latent actions, and a VAE decoder
        # that maps latent actions back to executable actions.
        if extract_dynamic_features is None or action_latent_encoder is None or action_latent_decoder is None:
            raise ValueError(
                "DCDP runner requires extract_dynamic_features, "
                "action_latent_encoder, and action_latent_decoder."
            )

        device = policy.device
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False

            # Keep the most recent visual frames for the dynamics extractor.
            # obs['image'] has an observation-history dimension, so -1 is the newest frame.
            latest_image = obs['image'][:, -1]
            obs_buffer = deque([latest_image] * history, maxlen=history)

            action_horizon = action_latent_encoder.action_horizon
            action_dim = action_latent_encoder.action_dim
            latent_dim = action_latent_encoder.latent_dim
            if self.n_action_steps % action_horizon != 0:
                raise ValueError(
                    "DCDP evaluation expects n_action_steps to be divisible by "
                    f"the VAE action_horizon. Got {self.n_action_steps} and {action_horizon}."
                )

            # The DCDP decoder consumes one dynamics window per decoded action chunk.
            dynamic_buffer = deque(
                [extract_dynamic_features(obs_buffer)] * action_horizon,
                maxlen=action_horizon)

            # Force a fresh policy prediction on the first loop iteration. After that,
            # index_action tracks which low-level action is being executed from the
            # currently decoded action chunk.
            index_action = self.n_action_steps - 1

            # Use the VAE checkpoint's action range so encoder normalization matches
            # decoder denormalization.
            action_min_torch = action_latent_decoder.action_min.to(device)
            action_max_torch = action_latent_decoder.action_max.to(device)

            while not done:
                # Update dynamics context with the latest frame before decoding the next action.
                obs_buffer.append(obs['image'][:, -1])
                dynamic_buffer.append(extract_dynamic_features(obs_buffer))

                if index_action == self.n_action_steps - 1:
                    # device transfer once with correct dtype
                    obs_dict = dict_apply(dict(obs), 
                        lambda x: torch.from_numpy(x).to(device=device, dtype=torch.float32))
                    
                    # run policy
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)

                        # Split the policy action sequence into VAE-sized chunks and encode
                        # each chunk into one latent action.
                        action_torch = action_dict['action']
                        if (
                            action_torch.ndim != 3
                            or action_torch.shape[-1] != action_dim
                            or action_torch.shape[1] != self.n_action_steps
                            or action_torch.shape[1] % action_horizon != 0
                        ):
                            raise ValueError(
                                "DCDP evaluation expects policy actions with shape "
                                f"(B, {self.n_action_steps}, {action_dim}) and an action "
                                f"horizon divisible by {action_horizon}, got {tuple(action_torch.shape)}."
                            )
                        num_chunks = action_torch.shape[1] // action_horizon
                        action_torch = action_torch.reshape(-1, action_horizon, action_dim)
                        
                        # Normalize actions to the VAE training range before latent encoding.
                        action_norm = 2 * (action_torch - action_min_torch) / (action_max_torch - action_min_torch + 1e-8) - 1
                        
                        # Encode action chunks into latent actions per environment.
                        latent_action = action_latent_encoder(action_norm).reshape(-1, num_chunks, latent_dim)
                    index_action = -1

                # Decode the current latent action into an action chunk using the
                # latest dynamic-feature window, then execute only the next low-level action.
                action_tensor = action_latent_decoder(latent_action, index_action + 1, dynamic_buffer)
                index_action += 1
                chunk_offset = index_action % action_horizon
                next_action = action_tensor[:, chunk_offset:chunk_offset + 1, :].detach().cpu().numpy()

                # step env
                obs, reward, done, info = env.step(next_action)
                done = np.all(done)
                # update pbar
                pbar.update(1)  # update by 1 step per iteration
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
