import gobang_game
import gym

env = gym.make('Gobang9x9-v1')#('Gobang9x9-v0')

#env.reset()
'''
a,b,c,d = env.step(15)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(33)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(10)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(66)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(44)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(36)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(16)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(34)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(11)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(67)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(45)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(37)
print(a)
input("Press enter to continue")
a,b,c,d = env.step(54)
print(a)
input("Press enter to continue")
exit()
'''
# play a game

env.reset()
for i in range(90):
    env.render()
    action = int(input("input:"))#env.action_space.sample() # sample without replacement
    #print(action)
    observation, reward, done, info = env.step(action)
    if done:
        print ("Game is Over")
        if info['winner'] is not None:
            print("The Winner is "+"Black" if info['winner']==1 else "White")
        else:
            print("Tie !")
        break