import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

print("=" * 50)
print("TREINAMENTO DE IA COM STABLE BASELINES 3")
print("=" * 50)

# Criar ambiente vetorizado (mais rápido para treinamento)
print("\n1. Criando ambiente CartPole...")
env = make_vec_env("CartPole-v1", n_envs=4)

# Criar modelo PPO (Proximal Policy Optimization)
print("2. Criando modelo PPO...")
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)

# Treinar o modelo
print("\n3. Treinando o modelo por 50.000 passos...")
model.learn(total_timesteps=50000)

# Salvar o modelo treinado
print("\n4. Salvando o modelo...")
model.save("cartpole_ppo_model")
print("Modelo salvo como 'cartpole_ppo_model'")

# Testar o modelo treinado
print("\n5. Testando o modelo treinado...")
test_env = gym.make("CartPole-v1", render_mode="human")

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
        
        if steps > 500:  # Limite de segurança
            break
    
    rewards_list.append(episode_reward)
    print(f"Episódio de teste {episode + 1}: Reward = {episode_reward:.0f}, Steps = {steps}")

test_env.close()

# Estatísticas finais
print("\n" + "=" * 50)
print("ESTATÍSTICAS FINAIS")
print("=" * 50)
print(f"Recompensa média nos 5 testes: {np.mean(rewards_list):.2f}")
print(f"Recompensa máxima: {np.max(rewards_list):.0f}")
print(f"Recompensa mínima: {np.min(rewards_list):.0f}")
print("=" * 50)
