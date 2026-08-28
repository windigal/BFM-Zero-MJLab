"""Expert trajectory loading helpers shared by MJLab training."""

import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.envs.env_utils.history_handler import HistoryHandler as HVHistoryHandler
from humanoidverse.envs.motion_observations import compute_humanoid_observations_max
from humanoidverse.utils.reference_observations import reference_base_ang_vel
from humanoidverse.utils.torch_utils import quat_rotate_inverse

from ..buffers.trajectory import TrajectoryDictBuffer


def load_expert_trajectories_from_motion_lib(env, agent_cfg, device="cpu", add_history_noaction: bool = False):
    """Load expert trajectories directly from an environment motion library."""
    episodes = []
    file_names = []
    history_handler = HVHistoryHandler(1, env.config.obs.obs_auxiliary, env.config.obs.obs_dims, env.device)
    history_config = env.config.obs.obs_auxiliary["history_actor"]
    motion_lib = env._motion_lib
    num_unique_motions = int(motion_lib._num_unique_motions)
    global_motion_keys = motion_lib._motion_data_keys.tolist()
    chunk_size = max(int(env.num_envs), 1)
    for chunk_start in range(0, num_unique_motions, chunk_size):
        global_ids = list(range(chunk_start, min(chunk_start + chunk_size, num_unique_motions)))
        motion_lib.all_motions_loaded = False
        motion_lib.load_motions(random_sample=False, motion_ids=global_ids)
        for local_id, global_id in enumerate(global_ids):
            motion_times = torch.arange(int(np.ceil((motion_lib._motion_lengths[local_id] / env.dt).cpu()))).to(env.device) * env.dt
            motion_id = torch.full((motion_times.shape[0],), local_id, dtype=torch.long, device=env.device)
            motion_res = motion_lib.get_motion_state(motion_id, motion_times)
            file_names.append(global_motion_keys[global_id])

            ref_body_pos = motion_res["rg_pos_t"]
            ref_body_rots = motion_res["rg_rot_t"]
            ref_body_vels = motion_res["body_vel_t"]
            ref_body_angular_vels = motion_res["body_ang_vel_t"]

            obs_dict = compute_humanoid_observations_max(
                ref_body_pos,
                ref_body_rots,
                ref_body_vels,
                ref_body_angular_vels,
                local_root_obs=True,
                root_height_obs=env.config.obs.root_height_obs,
            )
            max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)

            base_quat = ref_body_rots[:, 0]
            ref_dof_pos = motion_res["dof_pos"] - env.default_dof_pos[0]
            ref_dof_vel = motion_res["dof_vel"]
            ref_ang_vel = reference_base_ang_vel(env, base_quat, ref_body_angular_vels[:, 0])
            projected_gravity = quat_rotate_inverse(
                base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True
            )
            bogus_actions = ref_dof_pos * 0

            state = torch.cat([ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel], dim=-1)

            data = {
                "base_ang_vel": ref_ang_vel,
                "projected_gravity": projected_gravity,
                "dof_pos": ref_dof_pos,
                "dof_vel": ref_dof_vel,
            }

            if add_history_noaction:
                history_handler.reset([0])
                history_actor = []
                for ii in range(state.shape[0]):
                    history_tensors = []
                    for key in sorted(history_config.keys()):
                        if key not in ["action", "actions"]:
                            history_length = history_config[key]
                            history_tensor = history_handler.query(key)[:, :history_length]
                            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
                            history_tensors.append(history_tensor)
                    history_tensors = torch.cat(history_tensors, dim=1)
                    history_actor.append(history_tensors)

                    for key in history_handler.history.keys():
                        if key not in ["action", "actions"]:
                            history_handler.add(key, data[key][ii][None, ...])
                history_actor = torch.stack(history_actor, dim=0).squeeze(1)

            curr_motion_len = state.shape[0]
            minimum_length = int(agent_cfg.model.seq_length) + 1
            if curr_motion_len < minimum_length:
                raise ValueError(
                    f"Expert trajectory motion_id={global_id}, motion={global_motion_keys[global_id]!r} has "
                    f"{curr_motion_len} steps; at least {minimum_length} are required"
                )
            truncated = torch.zeros(curr_motion_len, dtype=bool, device=env.device)
            truncated[-1] = True

            assert state.shape[0] == curr_motion_len
            assert max_local_self_obs.shape[0] == curr_motion_len
            assert bogus_actions.shape[0] == curr_motion_len
            assert truncated.shape[0] == curr_motion_len
            if add_history_noaction:
                assert history_actor.shape[0] == curr_motion_len

            ep = {
                "observation": {
                    "state": state,
                    "last_action": bogus_actions,
                    "privileged_state": max_local_self_obs,
                },
                "terminated": torch.zeros(curr_motion_len, dtype=torch.bool, device=env.device),
                "truncated": truncated,
                "motion_id": torch.full((curr_motion_len,), global_id, dtype=torch.long, device=env.device),
            }
            if add_history_noaction:
                ep["observation"]["history_noaction"] = history_actor
            episodes.append(tree_map(lambda value: value.cpu() if torch.is_tensor(value) else value, ep))

    motion_lib.all_motions_loaded = False
    motion_lib.load_motions_for_training(max_num_seqs=env.num_envs)

    expert_buffer = TrajectoryDictBuffer(
        episodes=episodes,
        seq_length=agent_cfg.model.seq_length,
        device=device,
    )

    assert expert_buffer.storage["observation"]["state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["observation"]["last_action"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["observation"]["privileged_state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["terminated"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["motion_id"].shape[0] == expert_buffer.storage["truncated"].shape[0]

    expert_buffer.file_names = file_names
    return expert_buffer
