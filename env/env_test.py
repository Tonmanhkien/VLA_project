import gymnasium as gym
import metaworld
import time


env = gym.make("Meta-World/MT1", env_name="reach-v3", render_mode="human")

observation, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.01)
    print(f"Obs: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

env.close()