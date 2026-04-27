import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import time


def main():
    print("=" * 60)
    print("TREINAMENTO DQN - MountainCar")
    print("=" * 60)

    # Ambiente para treinamento (sem renderização para melhor performance)
    env = make_vec_env("MountainCar-v0", n_envs=4)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )

    print("\nTreinando por 100.000 passos...")
    model.learn(total_timesteps=100000)

    print("Salvando o modelo em 'mountaincar_dqn_sb3'...")
    model.save("mountaincar_dqn_sb3")
    env.close()

    print("\nTestando o modelo salvo em 5 episódios...")
    test_env = gym.make("MountainCar-v0", render_mode="human")
    rewards = []

    for episode in range(5):
        observation, info = test_env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            test_env.render()  # Renderiza o ambiente

        rewards.append(episode_reward)
        print(f"Episódio {episode + 1}: recompensa = {episode_reward:.2f}")
        time.sleep(0.5)  # Pausa entre episódios para visualizar melhor

    test_env.close()

    print("\n" + "=" * 60)
    print("RESULTADOS DE TESTE - MountainCar")
    print("=" * 60)
    print(f"Recompensa média: {np.mean(rewards):.2f}")
    print(f"Recompensa mínima: {np.min(rewards):.2f}")
    print(f"Recompensa máxima: {np.max(rewards):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
