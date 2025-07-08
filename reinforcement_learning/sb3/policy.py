# Copyright (c) 2022-2025, LAJi.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from typing import Any, Optional, Tuple, Union

from gymnasium import spaces
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device


class DACCNetwork(nn.Module):
    """
    Distributed Actor Centralized Critic (DACC) network. It receives as input the concatenated features of all agents extracted by the features extractor and outputs a concatenated latent representation for the distributed policy and a centralized value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param central_feature_dim: Dimension of the concatenated feature vector.
    :param num_agents: Number of agents.
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        central_feature_dim: int,
        num_agents: int,
        net_arch: Union[list[int], dict[str, list[int]]],
        activation_fn: type[nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        self.num_agents = num_agents
        assert central_feature_dim % num_agents == 0, "central_feature_dim must be divisible by num_agents #^#"
        self.distributed_feature_dim = central_feature_dim // num_agents

        device = get_device(device)
        policy_net: list[nn.Module] = []
        value_net: list[nn.Module] = []
        last_layer_dim_pi = self.distributed_feature_dim
        last_layer_dim_vf = central_feature_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specified, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch

        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi * num_agents
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # Shape of features: [batch_size, central_feature_dim]
        batch_size = features.shape[0]
        # Split into: [batch_size, num_agents, per_agent_dim]
        split_features = features.view(batch_size, self.num_agents, self.distributed_feature_dim)

        # Process each agent separately through shared policy_net
        latents = []
        for i in range(self.num_agents):
            agent_feature = split_features[:, i, :]
            agent_latent = self.policy_net(agent_feature)
            latents.append(agent_latent)

        # Concatenate along feature dimension: [batch_size, num_agents * agent_latent_dim]
        return th.cat(latents, dim=1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class DistributedActorCentralizedCriticPolicy(ActorCriticPolicy):
    """
    Multi-agent policy class for single-agent actor-critic algorithms (has both distributed policy and centralized value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param num_agents: Number of agents.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        num_agents: int = 1,
    ):
        self.num_agents = num_agents

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = DACCNetwork(
            self.features_dim,
            self.num_agents,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            _, self.log_std = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi, log_std_init=self.log_std_init)
            # Distributed action net
            self.action_net = nn.Linear(latent_dim_pi // self.num_agents, self.action_dist.action_dim // self.num_agents)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if isinstance(self.action_dist, DiagGaussianDistribution):
            batch_size = latent_pi.shape[0]
            latent_dim_pi = self.mlp_extractor.latent_dim_pi
            split_latent = latent_pi.view(batch_size, self.num_agents, latent_dim_pi // self.num_agents)

            mean_actions = []
            for i in range(self.num_agents):
                agent_latent = split_latent[:, i, :]
                agent_mean_action = self.action_net(agent_latent)
                mean_actions.append(agent_mean_action)

            mean_actions = th.cat(mean_actions, dim=1)
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        else:
            mean_actions = self.action_net(latent_pi)
            if isinstance(self.action_dist, CategoricalDistribution):
                # Here mean_actions are the logits before the softmax
                return self.action_dist.proba_distribution(action_logits=mean_actions)
            elif isinstance(self.action_dist, MultiCategoricalDistribution):
                # Here mean_actions are the flattened logits
                return self.action_dist.proba_distribution(action_logits=mean_actions)
            elif isinstance(self.action_dist, BernoulliDistribution):
                # Here mean_actions are the logits (before rounding to get the binary actions)
                return self.action_dist.proba_distribution(action_logits=mean_actions)
            elif isinstance(self.action_dist, StateDependentNoiseDistribution):
                return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
            else:
                raise ValueError("Invalid action distribution")
