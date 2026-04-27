import sys
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import numpy as np


def create_vec_env(env_id, n_envs=4):
    try:
        return make_vec_env(env_id, n_envs=n_envs)
    except Exception as ex:
        print(f"Erro ao criar ambiente {env_id}: {ex}")
        print("Instale a dependência necessária com:")
        print('pip install "gymnasium[box2d]"')
        sys.exit(1)


def create_env(env_id):
    try:
        return gym.make(env_id)
    except Exception as ex:
        print(f"Erro ao criar ambiente {env_id}: {ex}")
        print("Instale a dependência necessária com:")
        print('  pip install "gymnasium[box2d]"')
        sys.exit(1)


def main():
    print("=" * 60)
    print("TREINAMENTO DQN - LunarLander")
    print("=" * 60)

    env_id = "LunarLander-v3"
    env = create_vec_env(env_id, n_envs=4)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )

    print("\nTreinando por 150.000 passos...")
    model.learn(total_timesteps=150000)

    print("Salvando o modelo em 'lunarlander_dqn_sb3'...")
    model.save("lunarlander_dqn_sb3")
    env.close()

    print("\nTestando o modelo salvo em 5 episódios...")
    test_env = create_env(env_id)
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

        rewards.append(episode_reward)
        print(f"Episódio {episode + 1}: recompensa = {episode_reward:.2f}")

    test_env.close()

    print("\n" + "=" * 60)
    print("RESULTADOS DE TESTE - LunarLander")
    print("=" * 60)
    print(f"Recompensa média: {np.mean(rewards):.2f}")
    print(f"Recompensa mínima: {np.min(rewards):.2f}")
    print(f"Recompensa máxima: {np.max(rewards):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
