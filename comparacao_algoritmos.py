import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import time

print("=" * 60)
print("COMPARAÇÃO DE ALGORITMOS - STABLE BASELINES 3")
print("=" * 60)

# Configurações
algorithms = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN
}

continuous_algorithms = {
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG
}

# Ambiente discreto (CartPole)
print("\n" + "=" * 40)
print("TESTANDO ALGORITMOS EM CARTPOLE (DISCRETO)")
print("=" * 40)

cartpole_results = {}

for name, algorithm in algorithms.items():
    print(f"\n--- TREINANDO {name} ---")

    # Criar ambiente
    env = make_vec_env("CartPole-v1", n_envs=4)

    # Configurar parâmetros específicos para cada algoritmo
    if name == "DQN":
        model = algorithm("MlpPolicy", env, verbose=0, learning_rate=0.001,
                         buffer_size=100000, learning_starts=1000, batch_size=32,
                         tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1,
                         target_update_interval=1000, exploration_fraction=0.1,
                         exploration_initial_eps=1.0, exploration_final_eps=0.05)
    else:
        model = algorithm("MlpPolicy", env, verbose=0, learning_rate=0.001)

    # Treinar
    start_time = time.time()
    model.learn(total_timesteps=25000)  # Menos passos para comparação rápida
    train_time = time.time() - start_time

    # Testar
    test_env = gym.make("CartPole-v1")
    rewards = []

    for _ in range(3):  # 3 testes por algoritmo
        obs, _ = test_env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0

        while not (done or truncated) and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = test_env.step(action)
            episode_reward += reward
            steps += 1

        rewards.append(episode_reward)

    test_env.close()
    cartpole_results[name] = {
        'mean_reward': np.mean(rewards),
        'max_reward': np.max(rewards),
        'train_time': train_time
    }

    print(f"{name}: Média = {np.mean(rewards):.1f}, Máx = {np.max(rewards):.0f}, Tempo = {train_time:.1f}s")

# Ambiente contínuo (Pendulum)
print("\n" + "=" * 40)
print("TESTANDO ALGORITMOS EM PENDULUM (CONTÍNUO)")
print("=" * 40)

pendulum_results = {}

for name, algorithm in continuous_algorithms.items():
    print(f"\n--- TREINANDO {name} ---")

    # Criar ambiente
    env = make_vec_env("Pendulum-v1", n_envs=4)

    # Configurar parâmetros específicos
    if name == "SAC":
        model = algorithm("MlpPolicy", env, verbose=0, learning_rate=0.001,
                         buffer_size=100000, learning_starts=1000, batch_size=256,
                         tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
                         ent_coef='auto', target_update_interval=1)
    elif name == "TD3":
        model = algorithm("MlpPolicy", env, verbose=0, learning_rate=0.001,
                         buffer_size=100000, learning_starts=1000, batch_size=256,
                         tau=0.005, gamma=0.99, train_freq=(1, "episode"),
                         gradient_steps=-1, action_noise=None, target_policy_noise=0.2,
                         target_noise_clip=0.5, policy_delay=2)
    else:  # DDPG
        model = algorithm("MlpPolicy", env, verbose=0, learning_rate=0.001,
                         buffer_size=100000, learning_starts=1000, batch_size=256,
                         tau=0.005, gamma=0.99, train_freq=(1, "episode"),
                         gradient_steps=-1, action_noise=None)

    # Treinar
    start_time = time.time()
    model.learn(total_timesteps=25000)
    train_time = time.time() - start_time

    # Testar
    test_env = gym.make("Pendulum-v1")
    rewards = []

    for _ in range(3):
        obs, _ = test_env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0

        while not (done or truncated) and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = test_env.step(action)
            episode_reward += reward
            steps += 1

        rewards.append(episode_reward)

    test_env.close()
    pendulum_results[name] = {
        'mean_reward': np.mean(rewards),
        'max_reward': np.max(rewards),
        'train_time': train_time
    }

    print(f"{name}: Média = {np.mean(rewards):.2f}, Máx = {np.max(rewards):.2f}, Tempo = {train_time:.1f}s")

# Resultados finais
print("\n" + "=" * 60)
print("RESUMO DOS RESULTADOS")
print("=" * 60)

print("\nCARTPOLE (Objetivo: 500 pontos):")
for name, results in cartpole_results.items():
    print(f"{name:4}: Média={results['mean_reward']:6.1f}, Máx={results['max_reward']:3.0f}, Tempo={results['train_time']:4.1f}s")

print("\nPENDULUM (Objetivo: ~0 pontos, recompensa negativa):")
for name, results in pendulum_results.items():
    print(f"{name:4}: Média={results['mean_reward']:6.2f}, Máx={results['max_reward']:6.2f}, Tempo={results['train_time']:4.1f}s")

print("\n" + "=" * 60)
print("CONCLUSÃO:")
print("- PPO e A2C são bons para ambientes discretos")
print("- SAC e TD3 geralmente performam melhor em contínuos")
print("- DQN é específico para ações discretas")
print("- DDPG é o predecessor do TD3")
print("=" * 60)