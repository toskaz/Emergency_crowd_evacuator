import random

from controls.evacuee_rl import QLearningFollowController
from controls.scripted_drone import ScriptedDroneController
from simulation import Simulation
from training.scenario import place_random_scenario
from utils.config import q_table_path


EPISODES = 600
MAX_STEPS = 300
EVACUEE_COUNT = 20
START_EPSILON = 1.0
END_EPSILON = 0.05
REPORT_EVERY = 100


def run_episode(simulation, evacuees, drones):
    total_reward = 0.0
    for _ in range(MAX_STEPS):
        if not simulation.evacuees:
            break
        evacuee_actions = evacuees.get_actions(simulation)
        drone_actions = drones.get_actions(simulation)
        simulation.step(evacuee_actions, drone_actions)
        total_reward += evacuees.learn(simulation)
    return total_reward


def epsilon_for(episode):
    progress = episode / max(EPISODES - 1, 1)
    return START_EPSILON + (END_EPSILON - START_EPSILON) * progress


def train(seed=0):
    random.seed(seed)
    rng = random.Random(seed)
    simulation = Simulation()
    evacuees = QLearningFollowController(learning=True)
    drones = ScriptedDroneController()

    rewards = []
    evacuated = []
    for episode in range(EPISODES):
        evacuees.epsilon = epsilon_for(episode)
        evacuees.reset()
        drones.reset()
        place_random_scenario(simulation, rng, EVACUEE_COUNT)
        start_count = len(simulation.evacuees)
        rewards.append(run_episode(simulation, evacuees, drones))
        evacuated.append(simulation.evacuated_count / max(start_count, 1))

        if (episode + 1) % REPORT_EVERY == 0:
            average_reward = sum(rewards[-REPORT_EVERY:]) / REPORT_EVERY
            average_evacuated = 100 * sum(evacuated[-REPORT_EVERY:]) / REPORT_EVERY
            print(
                f"episode {episode + 1:5d}  epsilon {evacuees.epsilon:.2f}  "
                f"reward {average_reward:8.2f}  evacuated {average_evacuated:5.1f}%"
            )

    evacuees.save(q_table_path)
    print(f"saved {q_table_path}")


if __name__ == "__main__":
    train()
