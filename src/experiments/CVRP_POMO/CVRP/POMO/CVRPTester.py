
import torch

import os
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils.utils import *


class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore - 智能加载：支持3维和5维checkpoint
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        
        # 尝试加载完整模型
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"✓ 模型加载成功 (完整checkpoint)")
        except RuntimeError as e:
            self.logger.warning(f"⚠ 维度不匹配，尝试智能加载...")
            # 只加载匹配的层
            model_state = self.model.state_dict()
            checkpoint_state = checkpoint['model_state_dict']
            
            loaded_keys = []
            skipped_keys = []
            for key in model_state:
                if key in checkpoint_state:
                    if model_state[key].shape == checkpoint_state[key].shape:
                        model_state[key] = checkpoint_state[key]
                        loaded_keys.append(key)
                    else:
                        skipped_keys.append(key)
                        self.logger.info(f"  跳过 {key}: 模型{list(model_state[key].shape)} vs checkpoint{list(checkpoint_state[key].shape)}")
            
            self.model.load_state_dict(model_state)
            self.logger.info(f"✓ 智能加载完成: 加载{len(loaded_keys)}层, 跳过{len(skipped_keys)}层")

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, test_data=None):
        """Run testing. If test_data dict is provided, use it instead of loading from file."""
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        if test_data is not None:
            self.env.use_saved_problems_from_dict(test_data)
        elif self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        step_count = 0
        max_steps = 100  # 减少步数，快速诊断
        
        # 记录每一步的选择
        selected_nodes = []
        
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            
            # 记录选择的节点
            selected_nodes.append(selected[0, 0].item())  # 只记录第一个
            
            state, reward, done = self.env.step(selected)
            step_count += 1
            
            # 调试信息：每5步输出一次详细信息
            if step_count % 5 == 0:
                # 检查有多少节点可选（没有被 mask）
                valid_nodes = (state.ninf_mask[0, 0] == 0).sum().item()
                energy_str = f"{state.current_energy[0,0].item():.3f}" if state.current_energy is not None else 'N/A'
                self.logger.info(f"Step {step_count}: selected={selected[0,0].item()}, valid_nodes={valid_nodes}/{self.env.problem_size+1}, load={state.load[0,0].item():.3f}, time={state.current_time[0,0].item():.3f}, energy={energy_str}")
                
                # 输出前10个可选节点的索引
                if step_count <= 15:
                    valid_indices = (state.ninf_mask[0, 0] == 0).nonzero().squeeze().tolist()
                    if isinstance(valid_indices, int):
                        valid_indices = [valid_indices]
                    self.logger.info(f"  可选节点（前10个）: {valid_indices[:10]}")
            
            # 防止死循环
            if step_count > max_steps:
                self.logger.warning(f"⚠ 超过 {max_steps} 步，强制停止")
                self.logger.warning(f"选择的节点序列: {selected_nodes[:50]}")
                # 强制计算reward（即使没有完成）
                # 注意：env.batch_size已经包含了aug_factor，所以reward形状应该是正确的
                reward = -self.env._get_travel_distance()
                # shape: (batch_size * aug_factor, pomo_size)
                self.logger.info(f"强制计算的reward形状: {reward.shape}")
                done = True
                break

        # Return
        ###############################################
        if reward is None:
            self.logger.warning("⚠ reward为None，使用默认值")
            reward = torch.zeros(aug_factor * batch_size, self.env.pomo_size)
        
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()
