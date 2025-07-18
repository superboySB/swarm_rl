# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class SwarmVelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 100
    max_iterations = 100000
    save_interval = 20
    experiment_name = ""
    clip_actions = 1.0
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 1024, 1024, 512, 512, 256],
        critic_hidden_dims=[1024, 1024, 1024, 512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )


@configclass
class SwarmAccPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50
    max_iterations = 1000
    save_interval = 20
    experiment_name = ""
    clip_actions = 1.0
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 1024, 1024, 512, 512, 256],
        critic_hidden_dims=[1024, 1024, 1024, 512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )


@configclass
class SwarmBodyratePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 250
    max_iterations = 100000
    save_interval = 20
    experiment_name = ""
    clip_actions = 1.0
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 2048, 2048, 1024, 1024, 1024, 512],
        critic_hidden_dims=[2048, 2048, 2048, 1024, 1024, 1024, 512],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
