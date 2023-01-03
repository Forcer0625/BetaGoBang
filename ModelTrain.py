import gym
import gobang_game
import stable_baselines3
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from PolicyValueNetwork import BetaGobangCnnExtractor, BetaGobangPolicy

env = gym.make("SPGobang9x9-v1")

#policy = ActorCriticPolicy(observation_space=env.observation_space, action_space=env.action_space, lr_schedule=scheduler)
#value_fn = ValueFunction(input_shape=env.observation_space.shape[0], output_shape=1)
policy_kwargs = dict(
    features_extractor_class=BetaGobangCnnExtractor,
    #features_extractor_kwargs=dict(features_dim=128, num_features=7, boardLength=9,)
)
for i in range(10):
    print("{i} Generation".format(i=i))
    model = A2C.load('./selfplay/model', env=env, device='cuda')
    model.learn(total_timesteps=int(100), progress_bar=False)
    model.save('./selfplay/model')
