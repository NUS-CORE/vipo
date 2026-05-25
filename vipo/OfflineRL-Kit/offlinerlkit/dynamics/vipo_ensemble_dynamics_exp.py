import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
from offlinerlkit.nets import MLP


class ValueNet(nn.Module):
    """V(s) 网络，使用 OfflineRL-Kit 里的 MLP 做 backbone."""
    def __init__(self, obs_dim: int, hidden_dims: List[int], device: torch.device):
        super().__init__()
        self.backbone = MLP(input_dim=obs_dim, hidden_dims=hidden_dims)
        self.v_head = nn.Linear(self.backbone.output_dim, 1)
        self.device = device
        self.to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (..., obs_dim)
        x = self.backbone(obs)
        v = self.v_head(x)
        return v.squeeze(-1)  # (...,)


class EnsembleValueNet(nn.Module):
    """Vᵉ(s) 的 ensemble 版，每个 ensemble 成员一套参数."""
    def __init__(self, obs_dim: int, hidden_dims: List[int], num_ensemble: int, device: torch.device):
        super().__init__()
        self.n_ensemble = num_ensemble
        self.nets = nn.ModuleList(
            [ValueNet(obs_dim, hidden_dims, device) for _ in range(num_ensemble)]
        )
        self.device = device
        self.to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (E, B, obs_dim)
        return: (E, B)
        """
        assert obs.dim() == 3, f"expected (E,B,obs_dim), got {obs.shape}"
        E, B, _ = obs.shape
        values = []
        for e in range(E):
            v = self.nets[e](obs[e])  # (B,)
            values.append(v)
        return torch.stack(values, dim=0)  # (E, B)


def gaussian_diag_log_prob(sample: torch.Tensor,
                           mean: torch.Tensor,
                           logvar: torch.Tensor) -> torch.Tensor:
    """
    多元独立高斯 N(mean, diag(exp(logvar))) 的 log_prob(x)，按最后一维求和。
    输入形状统一为 (..., D)，输出 (...,).
    """
    var = torch.exp(logvar)
    diff = sample - mean
    # log N(x; μ, Σ) = -0.5 [ D log(2π) + log|Σ| + (x-μ)^T Σ^{-1} (x-μ) ]
    log_det = torch.sum(torch.log(var + 1e-8), dim=-1)
    mahalanobis = torch.sum(diff * diff / (var + 1e-8), dim=-1)
    D = sample.shape[-1]
    log_prob = -0.5 * (D * np.log(2 * np.pi) + log_det + mahalanobis)
    return log_prob


class VIPOEnsembleDynamics_EXP(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        # 新增参数：
        discount: float = 0.99,
        phi: float = 1e-4,
        value_hidden_dims: List[int] = (256, 256),
        value_lr: float = 3e-4,
        advantage_normalization: bool = False,
        value_target_tau: float = 0.005,
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode

        self.discount = discount
        self.phi = phi
        self.value_hidden_dims = list(value_hidden_dims)
        self.value_lr = value_lr
        self.advantage_normalization = advantage_normalization
        self.value_target_tau = value_target_tau

        # value 网络懒初始化（第一次看到 batch 的时候再建）
        self._value_initialized = False
        self.true_value_net: Optional[ValueNet] = None
        self.ensemble_value_net: Optional[EnsembleValueNet] = None
        self.true_value_target_net: Optional[ValueNet] = None
        self.ensemble_value_target_net: Optional[EnsembleValueNet] = None
        self.true_value_optim: Optional[torch.optim.Optimizer] = None
        self.ensemble_value_optim: Optional[torch.optim.Optimizer] = None

    # =========================
    #  rollout（和原来一样）
    # =========================
    @torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info

    @torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss

    # =========================
    #  数据准备（多返回 obss）
    # =========================
    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        terminals = None
        for key in ("terminals", "dones", "done"):
            if key in data:
                terminals = data[key]
                break
        if terminals is None:
            raise KeyError("dataset must contain 'terminals' or 'dones' for done masking")
        if "timeouts" in data:
            terminals = np.logical_or(terminals, data["timeouts"])
        terminals = np.asarray(terminals).reshape(-1).astype(np.float32)
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets, obss, terminals

    def _maybe_init_value_nets(self, obs_dim: int):
        if self._value_initialized:
            return
        device = self.model.device
        num_ensemble = self.model.num_ensemble

        self.true_value_net = ValueNet(obs_dim, self.value_hidden_dims, device=device)
        self.true_value_target_net = ValueNet(obs_dim, self.value_hidden_dims, device=device)
        self.ensemble_value_net = EnsembleValueNet(obs_dim, self.value_hidden_dims, num_ensemble, device=device)
        self.ensemble_value_target_net = EnsembleValueNet(obs_dim, self.value_hidden_dims, num_ensemble, device=device)

        self.true_value_optim = torch.optim.Adam(self.true_value_net.parameters(), lr=self.value_lr)
        self.ensemble_value_optim = torch.optim.Adam(self.ensemble_value_net.parameters(), lr=self.value_lr)
        for param in self.true_value_target_net.parameters():
            param.requires_grad = False
        for param in self.ensemble_value_target_net.parameters():
            param.requires_grad = False
        self._update_value_targets(tau=1.0)

        self._value_initialized = True

    @torch.no_grad()
    def _update_value_targets(self, tau: Optional[float] = None) -> None:
        if (
            self.true_value_target_net is None
            or self.true_value_net is None
            or self.ensemble_value_target_net is None
            or self.ensemble_value_net is None
        ):
            return
        tau = self.value_target_tau if tau is None else tau

        def soft_update(target_net: nn.Module, source_net: nn.Module):
            for target_param, param in zip(target_net.parameters(), source_net.parameters()):
                target_param.data.mul_(1 - tau).add_(param.data, alpha=tau)

        soft_update(self.true_value_target_net, self.true_value_net)
        soft_update(self.ensemble_value_target_net, self.ensemble_value_net)

    # =========================
    #  训练：外循环
    # =========================
    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        inputs, targets, obss, dones = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs = inputs[train_splits.indices]
        train_targets = targets[train_splits.indices]
        train_obss = obss[train_splits.indices]
        train_dones = dones[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        # 懒初始化 value nets（这里已经知道 obs_dim ）
        obs_dim = obss.shape[1]
        self._maybe_init_value_nets(obs_dim)

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics (VIPO):")
        while True:
            epoch += 1
            train_loss, train_stats = self.learn(
                train_inputs[data_idxes],
                train_targets[data_idxes],
                train_obss[data_idxes],
                train_dones[data_idxes],
                batch_size,
                logvar_loss_coef
            )
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()

            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            # 一些监控 value inconsistency 的指标
            logger.logkv("loss/vipo_value_loss", train_stats["value_loss"])
            logger.logkv("stats/final_advantage", train_stats["final_advantage"])
            logger.logkv("stats/pred_return", train_stats["pred_return"])
            logger.logkv("stats/log_prob", train_stats["log_prob"])
            logger.logkv("stats/r_hat_mean", train_stats["r_hat_mean"])
            logger.logkv("stats/v_model_tp1", train_stats["v_model_tp1"])
            logger.logkv("stats/value_gap_l1", train_stats["value_gap_l1"])
            logger.logkv("stats/value_gap_l2", train_stats["value_gap_l2"])

            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
    
    # =========================
    #  训练：单 epoch 内部（含 VIPO loss）
    # =========================
    def learn(
        self,
        inputs: np.ndarray,    # (E, N, D_in)
        targets: np.ndarray,   # (E, N, D_out)
        obss: np.ndarray,      # (E, N, obs_dim)
        dones: np.ndarray,     # (E, N)
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        self.true_value_net.train()
        self.ensemble_value_net.train()

        train_size = inputs.shape[1]
        losses = []

        # 方便复用
        device = self.model.device
        discount = self.discount

        total_value_loss = 0.0
        total_gap_l1 = 0.0
        total_gap_l2 = 0.0
        n_batches = 0

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            obss_batch    = obss[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            dones_batch   = dones[:, batch_num * batch_size:(batch_num + 1) * batch_size]

            # (E, B, D)
            inputs_batch = torch.as_tensor(inputs_batch, dtype=torch.float32, device=device)
            targets_batch = torch.as_tensor(targets_batch, dtype=torch.float32, device=device)
            obss_batch = torch.as_tensor(obss_batch, dtype=torch.float32, device=device)
            dones_batch = torch.as_tensor(dones_batch, dtype=torch.float32, device=device).clamp(0.0, 1.0)
            not_done = 1.0 - dones_batch

            # ========== 1) dynamics base loss ==========
            mean, logvar = self.model(inputs_batch)  # (E, B, D_out)
            inv_var = torch.exp(-logvar)
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))

            dyn_loss = mse_loss_inv.sum() + var_loss.sum()
            dyn_loss = dyn_loss + self.model.get_decay_loss()
            dyn_loss = dyn_loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            # ========== 2) 训练 value 网络（只用 detach 特征） ==========
            # 拆出 obs / delta_obs / reward_true
            delta_obs_true = targets_batch[..., :-1]          # (E,B,obs_dim)
            reward_true    = targets_batch[..., -1]           # (E,B)
            obs_t          = obss_batch                      # (E,B,obs_dim)
            obs_tp1_true   = obs_t + delta_obs_true          # 真正的下一个 obs（来自数据）

            # 让 value 的特征都 detach 掉，不影响 dynamics
            obs_t_det    = obs_t.detach().reshape(-1, obs_t.shape[-1])          # (E*B, obs_dim)
            obs_tp1_det  = obs_tp1_true.detach().reshape(-1, obs_t.shape[-1])   # (E*B, obs_dim)
            reward_true_det = reward_true.detach().reshape(-1)                  # (E*B,)

            # true value：V(s) 拟合 r_true + γ V(s')
            v_s   = self.true_value_net(obs_t_det)       # (E*B,)
            with torch.no_grad():
                v_tp1 = self.true_value_target_net(obs_tp1_det)
                done_flat = dones_batch.reshape(-1)
                target_v = reward_true_det + discount * (1.0 - done_flat) * v_tp1
            true_v_loss = torch.mean((v_s - target_v) ** 2)

            # ensemble value：Vᵉ(s) 拟合 r̂_mean + γ Vᵉ(ŝ')
            # dynamics 的预测 mean：diff_mean = mean[..., :-1]，r_hat_mean = mean[..., -1]
            diff_mean = mean[..., :-1].detach()
            r_hat_mean = mean[..., -1].detach()
            obs_tp1_hat = (obs_t + diff_mean).detach()      # (E,B,obs_dim)

            v_model_s = self.ensemble_value_net(obs_t.detach())        # (E,B)
            with torch.no_grad():
                v_model_tp1_target = self.ensemble_value_target_net(obs_tp1_hat)
                target_model_v = r_hat_mean + discount * not_done * v_model_tp1_target   # (E,B)
            ensemble_v_loss = torch.mean((v_model_s - target_model_v) ** 2)

            # 先清 value 的 grad，单独 backward
            self.true_value_optim.zero_grad(set_to_none=True)
            self.ensemble_value_optim.zero_grad(set_to_none=True)
            (true_v_loss + ensemble_v_loss).backward()
            self.true_value_optim.step()
            self.ensemble_value_optim.step()
            self._update_value_targets()

            # ========== 3) VIPO value regularizer（只更新 dynamics） ==========
            # advantage = (V_true - V_model) * predicted_return
            with torch.no_grad():
                # 重新算一遍（用最新的 value 参数）
                v_true = self.true_value_net(obs_t.reshape(-1, obs_t.shape[-1])).view_as(r_hat_mean)  # (E,B)
                v_model = self.ensemble_value_net(obs_t)                                             # (E,B)
                v_model_tp1 = self.ensemble_value_target_net(obs_tp1_hat)                            # (E,B)
                pred_return = r_hat_mean + discount * not_done * v_model_tp1                        # (E,B)
                advantage = (v_true - v_model) * pred_return                                        # (E,B)
                if self.advantage_normalization:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
                # 记录 value gap 的统计量
                value_gap_l1 = torch.mean(torch.abs(v_true - v_model)).item()
                value_gap_l2 = torch.mean((v_true - v_model) ** 2).sqrt().item()

            advantage = advantage.detach()

            # 在 sample 点上算 log_prob（采样 stop-grad，只对 mean/logvar 求梯度）
            with torch.no_grad():
                eps = torch.randn_like(mean)
                sample = mean + eps * torch.sqrt(torch.exp(logvar))  # (E,B,D_out)

            log_prob = gaussian_diag_log_prob(sample, mean, logvar)  # (E,B)
            value_loss = -(advantage * log_prob).mean()

            # dynamics 的最终 loss
            total_loss = dyn_loss + self.phi * value_loss

            # ========== 4) 更新 dynamics ==========
            self.optim.zero_grad(set_to_none=True)
            total_loss.backward()
            self.optim.step()

            # ===== 记录统计 =====
            losses.append(total_loss.item())
            total_value_loss += value_loss.item()
            total_gap_l1 += value_gap_l1
            total_gap_l2 += value_gap_l2
            n_batches += 1

        avg_loss = float(np.mean(losses))
        stats = dict(
            value_loss=total_value_loss / max(n_batches, 1),
            value_gap_l1=total_gap_l1 / max(n_batches, 1),
            value_gap_l2=total_gap_l2 / max(n_batches, 1),
            final_advantage=advantage.mean().item(),
            pred_return=pred_return.mean().item(),
            r_hat_mean= r_hat_mean.mean().item(),
            v_model_tp1= v_model_tp1.mean().item(),
            log_prob=log_prob.mean().item(),
        )
        return avg_loss, stats
    
    @torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        """保持和原来一致：只保存 dynamics + scaler，不保存 value nets."""
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
