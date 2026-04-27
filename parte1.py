import gymnasium as gym
import numpy as np

# Criar o ambiente (CartPole é um ambiente simples do Gymnasium)
env = gym.make("CartPole-v1", render_mode="human")

# Fazer o reset do ambiente
observation, info = env.reset(seed=42)

# Executar 10 episódios
for episode in range(10):
    observation, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    # Rodar o episódio até terminar
    while not (done or truncated):
        # Escolher uma ação aleatória
        action = env.action_space.sample()
        
        # Executar a ação no ambiente
        observation, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        # Limite de passos para evitar loops infinitos
        if steps > 500:
            break
    
    print(f"Episódio {episode + 1}: Reward = {total_reward}, Steps = {steps}")

# Fechar o ambiente
env.close()
print("Programa finalizado!")