import torch
import numpy as np
import gym
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

class ConvolutionBlock(nn.Module):

    def __init__(self, in_features, out_features, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.batchNorm = nn.BatchNorm2d(num_features=out_features, eps=1e-5, momentum=.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        return F.relu(x)


class ResidualBlock(nn.Module):

    def __init__(self, inputDim=128, outputDim=128):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.kernelSize = 3

        self.conv1 = nn.Conv2d(in_channels=self.inputDim, out_channels=self.outputDim,
                               kernel_size=self.kernelSize, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=outputDim, out_channels=outputDim,
                               kernel_size=self.kernelSize, stride=1, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(num_features=self.outputDim)
        self.batch_norm2 = nn.BatchNorm2d(num_features=self.outputDim)

    def forward(self, x):
        out1 = F.relu(self.batch_norm1(self.conv1(x)))
        out2 = self.batch_norm2(self.conv2(out1))
        return F.relu(out2 + x)

class BetaGobangCnnExtractor(BaseFeaturesExtractor):
    '''
        policy_kwargs = dict(
        features_extractor_class=BetaGobangCnnExtractor,
        features_extractor_kwargs=dict(features_dim=128, num_features=7, boardLength=9, use_gpu=True),
    )'''

    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    
    def __init__(self, observation_space: gym.spaces.Box,
                features_dim: int = 128,
                ):
        super(BetaGobangCnnExtractor, self).__init__(observation_space, features_dim)

        self.boardLength = 9#boardLength
        self.num_feature = 6+2#num_feature
        #self.features_dim  = features_dim
        #self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.conv = ConvolutionBlock(self.num_feature, features_dim, kernel_size=3, padding=1)
        self.residues = nn.Sequential(*[ResidualBlock(features_dim, features_dim) for i in range(4)])
       

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.conv(observations)
        return self.residues(x)



class PolicyNet(nn.Module):

    def __init__(self, in_features=128, boardLength=9):
        super().__init__()
        self.in_features = in_features
        self.boardLength = boardLength
        self.conv = ConvolutionBlock(in_features=self.in_features,
                                    out_features=2,
                                    kernel_size=1)
        self.linear = nn.Linear(in_features=2*self.boardLength**2,
                            out_features=self.boardLength**2+1)

    def forward_actor(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x.flatten(1))
        return F.log_softmax(input=x, dim=1)


class ValueNet(nn.Module):

    def __init__(self, in_features=128, boardLength=9):
        super().__init__()
        self.in_features = in_features
        self.boardLength = boardLength
        self.kernel_size = 1
        self.conv = ConvolutionBlock(in_features=in_features,
                                    out_features=1, # 輸出估計值
                                    kernel_size=self.kernel_size)
        self.model = nn.Sequential(
            nn.Linear(in_features=boardLength**2, out_features=self.in_features),
            nn.ReLU(),
            nn.Linear(in_features=self.in_features, out_features=1),
            nn.Tanh()
        )

    def forward_critic(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv(x)
        x = self.model(x.flatten(1))
        return x

class BetaGobangPolicyValueNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 9*9+1,
        last_layer_dim_vf: int = 1,
    ):
        super(BetaGobangPolicyValueNetwork, self).__init__()
 
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Policy network
        self.policy_net = PolicyNet(in_features=feature_dim, boardLength=9)
        # Value network
        self.value_net = ValueNet(in_features=feature_dim, boardLength=9)


    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class BetaGobangPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(BetaGobangPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = BetaGobangPolicyValueNetwork(self.features_dim)


class PolicyValueNetwork(nn.Module):

    def __init__(self, boardLength=9, num_feature=7):
        super().__init__()
        self.boardLength = boardLength
        self.num_feature = num_feature
        self.num_dim  = 128
        #self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.conv = ConvolutionBlock(self.num_feature, self.num_dim, 3, padding=1)
        self.residues = nn.Sequential(*[ResidualBlock(self.num_dim, self.num_dim) for i in range(4)])
        self.policy_net = PolicyNet(self.num_dim, self.boardLength)
        self.value_net = ValueNet(self.num_dim, self.boardLength)

    def forward(self, x):
        x = self.conv(x)
        x = self.residues(x)
        vec_p = self.policy_net(x)
        estimate_value = self.value_net(x)
        return vec_p, estimate_value

    def predict(self, observation: np.ndarray):
        """ 獲取當前局面上所有可用 `action` 和他對應的先驗概率 `P(s, a)`，以及局面的 `value`
        Parameters
        ----------
        chess_board
            棋盤
        Returns
        -------
        probs: `np.ndarray` of shape `(len(chess_board.available_actions), )`
            當前局面上所有可用 `action` 對應的先驗概率 `P(s, a)`
        value: float
            當前局面的估值
        """
        feature_planes = observation.get_feature_planes().to(self.device)
        feature_planes.unsqueeze_(0)
        vec_p, estimate_value = self(feature_planes)

        probs = torch.exp(vec_p).flatten()

        # 只取可行的落點
        if self.use_gpu:
            probs = probs[observation.available_actions].cpu().detach().numpy()
        else:
            probs = probs[observation.available_actions].detach().numpy()

        return probs, estimate_value[0].item()