import argparse
import os
import sys
import json
import gymnasium as gym
import numpy as np
import metaworld
from metaworld.policies import ENV_POLICY_MAP

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.tokenizer import SimpleTokenizer 

def parse_args():
    parse = argparse.ArgumentParser(description="Collect data from Meta-World environments.")
    parse.add_argument("--env_name", type=str, default="reach-v3", help="Name of the Meta-World environment.")
    parse.add_argument("--camera_view", type=str, default="top_view", help="corner, corner2, corner3, corner4, topview, behindGripper, gripperPOV")
    parse.add_argument("--seed", type=int, default=42)
    parse.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to collect.")
    parse.add_argument("--max_steps", type=int, default=150, help="Maximum steps per episode.")
    parse.add_argument("--instruction", type=str, default="push object to goal")
    parse.add_argument("--output_dir", type=str, default="./data", help="Directory to save collected data.")
    return parse.parse_args()


def extract_state(obs):
    return np.asarray(obs, dtype=np.float32).ravel()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    env = gym.make("Meta-World/MT1", env_name=args.env_name, seed=args.seed, render_mode="rgb_array", camera_name=args.camera_view)

    images, states, actions, texts = [], [], [], []

    instruction = args.instruction

    policy = ENV_POLICY_MAP[args.env_name]()

    for episode in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        step = 0

        while not done and step < args.max_steps:
            action = policy.get_action(obs)

            img = env.render()  # (H, W, 3)
            state = extract_state(obs) # (state_dim,)

            images.append(img.copy())
            states.append(state.copy())
            actions.append(action.copy())
            texts.append(instruction)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        print(f"Collected episode {episode + 1}/{args.num_episodes}")

    env.close()
    print("Stacking images")
    images = np.stack(images)  # (N, H, W, 3)
    print("Stacking states and actions")
    states = np.stack(states)  # (N, state_dim)
    actions = np.stack(actions)  # (N, action_dim)

    tokenizer = SimpleTokenizer(vocab=None)
    tokenizer.build_from_texts(texts)
    text_ids_list = [tokenizer.encode(text) for text in texts]
    max_len = max(len(ids) for ids in text_ids_list)
    text_ids = np.zeros((len(texts), max_len), dtype=np.int32)

    for i, ids in enumerate(text_ids_list):
        text_ids[i, :len(ids)] = np.array(ids, dtype=np.int32)

    np.savez_compressed(
        args.output_dir,
        images=images,
        states=states,
        actions=actions,
        text_ids=text_ids,
        vocab=tokenizer.vocab,
    )
    print(f"Data saved to {args.output_dir}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print("images shape:", images.shape)
    print("states shape:", states.shape)
    print("actions shape:", actions.shape)
    print("text_ids shape:", text_ids.shape)

if __name__ == "__main__":
    main()
        



