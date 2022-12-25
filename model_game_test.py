import gym
import gobang_game
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('Gobang9x9-v1')
model = A2C.load("./selfplay/model", env=env)

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#print(f'Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}')

obs = env.reset()
for _ in range(400):
    #print("sample:", env.action_space.sample())
    env.render()
    action, _state = model.predict(obs, deterministic=False)#int(input("=> :"))#env.action_space.sample() # sample without replacement
    print("model action:", action)
    obs, reward, done, info = env.step(action)
    
    if done:
        print ("Game is Over")
        env.render()
        if info['winner'] is not None:
            print("The Winner is ", "Black" if info['winner']==1 else "White")
        else:
            print("Tie !")
        break

    env.render()
    action = int(input("your action:"))
    obs, reward, done, info = env.step(action)

    if done:
        print ("Game is Over")
        env.render()
        if info['winner'] is not None:
            print("The Winner is "+"Black" if info['winner']==1 else "White")
        else:
            print("Tie !")
        break