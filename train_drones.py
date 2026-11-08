import random

from training.drone_env import DroneNavigationEnv
from utils.config import drone_model_path


EPISODES = 2000
MAX_EPISODE_STEPS = 400
EPSILON_START = 0.25
EPSILON_END = 0.03
EPSILON_DECAY_STEPS = 100_000
REPORT_EVERY = 100


def train(seed=0):
    random.seed(seed)
    env = DroneNavigationEnv(max_episode_steps=MAX_EPISODE_STEPS)
    controller = env.controller
    total_steps = 0
    episode_rewards = []
    episode_successes = []

    for episode in range(EPISODES):
        observation, _ = env.reset(seed=seed if episode == 0 else None)
        episode_reward = 0.0
        success = False

        while True:
            progress = min(total_steps / EPSILON_DECAY_STEPS, 1.0)
            controller.epsilon = EPSILON_START + progress * (
                EPSILON_END - EPSILON_START
            )
            action = env.select_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            total_steps += 1

            if terminated or truncated:
                success = terminated
                break

        episode_rewards.append(episode_reward)
        episode_successes.append(float(success))
        if (episode + 1) % REPORT_EVERY == 0:
            recent_rewards = episode_rewards[-REPORT_EVERY:]
            recent_successes = episode_successes[-REPORT_EVERY:]
            print(
                f"episode {episode + 1:5d}  "
                f"reward {sum(recent_rewards) / len(recent_rewards):8.2f}  "
                f"success {100 * sum(recent_successes) / len(recent_successes):5.1f}%  "
                f"epsilon {controller.epsilon:.3f}"
            )

    controller.save(drone_model_path)
    env.close()
    print(f"saved {drone_model_path}")


if __name__ == "__main__":
    train()