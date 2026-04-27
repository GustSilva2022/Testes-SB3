import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

print("=" * 50)
print("TREINAMENTO COM DDPG (Deep Deterministic Policy Gradient)")
print("=" * 50)

# Criar ambiente Pendulum (ambiente contínuo)
print("\n1. Criando ambiente Pendulum...")
env = make_vec_env("Pendulum-v1", n_envs=4)

# Criar modelo DDPG
print("2. Criando modelo DDPG...")
model = DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001,
             buffer_size=100000, learning_starts=1000, batch_size=256,
             tau=0.005, gamma=0.99, train_freq=(1, "episode"),
             gradient_steps=-1, action_noise=None)

# Treinar o modelo
print("\n3. Treinando o modelo por 50.000 passos...")
model.learn(total_timesteps=50000)

# Salvar o modelo treinado
print("\n4. Salvando o modelo...")
model.save("pendulum_ddpg_model")
print("Modelo salvo como 'pendulum_ddpg_model'")

# Testar o modelo treinado
print("\n5. Testando o modelo treinado...")
test_env = gym.make("Pendulum-v1", render_mode="human")

episodes_to_test = 5
rewards_list = []

for episode in range(episodes_to_test):
    observation, info = test_env.reset()
    done = False
    truncated = False
    episode_reward = 0
    steps = 0

    while not (done or truncated):
        # Usar o modelo treinado para prever ação
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, truncated, info = test_env.step(action)

        episode_reward += reward
        steps += 1

        if steps > 200:  # Limite de segurança para Pendulum
            break

    rewards_list.append(episode_reward)
    print(f"Episódio de teste {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

test_env.close()

# Estatísticas finais
print("\n" + "=" * 50)
print("ESTATÍSTICAS FINAIS - DDPG")
print("=" * 50)
print(f"Recompensa média nos 5 testes: {np.mean(rewards_list):.2f}")
print(f"Recompensa máxima: {np.max(rewards_list):.2f}")
print(f"Recompensa mínima: {np.min(rewards_list):.2f}")
print("=" * 50)