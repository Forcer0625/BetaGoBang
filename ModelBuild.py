import gym
import gobang_game
import stable_baselines3
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from PolicyValueNetwork import BetaGobangCnnExtractor, BetaGobangPolicy

env = gym.make("Gobang9x9-v0")

#policy = ActorCriticPolicy(observation_space=env.observation_space, action_space=env.action_space, lr_schedule=scheduler)
#value_fn = ValueFunction(input_shape=env.observation_space.shape[0], output_shape=1)
policy_kwargs = dict(
    features_extractor_class=BetaGobangCnnExtractor,
    #features_extractor_kwargs=dict(features_dim=128, num_features=7, boardLength=9,)
)


model = A2C(
    env=env,
    policy=BetaGobangPolicy,
    n_steps=5,
    #batch_size=32,
    gamma=0,
    verbose=1,
    device='cuda',
    policy_kwargs=policy_kwargs,
)
model.learn(total_timesteps=int(10), progress_bar=False)
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

#print(f'Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')

model.save('./selfplay/model')
