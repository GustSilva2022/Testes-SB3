# 🚀 Guia Completo: Algoritmos de Aprendizado por Reforço

Este repositório contém implementações práticas dos principais algoritmos do **Stable Baselines 3** usando **Gymnasium**.

## 📁 Arquivos Disponíveis

### Arquivos Individuais por Algoritmo

| Arquivo | Algoritmo | Ambiente | Tipo |
|---------|-----------|----------|------|
| `parte1.py` | **Ações Aleatórias** | CartPole | Baseline |
| `parte2.py` | **PPO** | CartPole | On-policy |
| `a2c_cartpole.py` | **A2C** | CartPole | On-policy |
| `dqn_cartpole.py` | **DQN** | CartPole | Off-policy |
| `sac_pendulum.py` | **SAC** | Pendulum | Off-policy |
| `td3_pendulum.py` | **TD3** | Pendulum | Off-policy |
| `ddpg_pendulum.py` | **DDPG** | Pendulum | Off-policy |
| `comparacao_algoritmos.py` | **Comparação** | CartPole + Pendulum | Todos |

## 🧠 Algoritmos Explicados

### 1. **PPO (Proximal Policy Optimization)**
- **Tipo**: On-policy
- **Ambiente**: CartPole (discreto)
- **Características**:
  - Estável e confiável
  - Bom equilíbrio entre exploração e exploração
  - Usa clipping para limitar mudanças na política
- **Pontos Fortes**: Robusto, converge bem, amplamente usado

### 2. **A2C (Advantage Actor-Critic)**
- **Tipo**: On-policy
- **Ambiente**: CartPole (discreto)
- **Características**:
  - Usa vantagem (advantage) para reduzir variância
  - Paraleliza múltiplos ambientes
  - Simples de implementar
- **Pontos Fortes**: Eficiente, bom para exploração

### 3. **DQN (Deep Q-Network)**
- **Tipo**: Off-policy
- **Ambiente**: CartPole (discreto)
- **Características**:
  - Usa replay buffer
  - Target network para estabilidade
  - ε-greedy para exploração
- **Pontos Fortes**: Bom para problemas discretos, inspirado no AlphaGo

### 4. **SAC (Soft Actor-Critic)**
- **Tipo**: Off-policy
- **Ambiente**: Pendulum (contínuo)
- **Características**:
  - Maximiza entropia (exploração)
  - Dois Q-networks para reduzir overestimation
  - Aprende temperatura automaticamente
- **Pontos Fortes**: Robusto, bom em contínuos, exploração natural

### 5. **TD3 (Twin Delayed DDPG)**
- **Tipo**: Off-policy
- **Ambiente**: Pendulum (contínuo)
- **Características**:
  - Dois Q-networks (Twin)
  - Delayed policy updates
  - Target policy smoothing
- **Pontos Fortes**: Reduz overestimation, melhor que DDPG

### 6. **DDPG (Deep Deterministic Policy Gradient)**
- **Tipo**: Off-policy
- **Ambiente**: Pendulum (contínuo)
- **Características**:
  - Determinístico (não estocástico)
  - Actor-Critic architecture
  - Usa replay buffer
- **Pontos Fortes**: Bom para ações contínuas, predecessor do TD3

## 🎯 Ambientes Utilizados

### **CartPole-v1** (Discreto)
- **Objetivo**: Equilibrar bastão por 500 passos
- **Ações**: 2 (esquerda, direita)
- **Estado**: 4 valores (posição, velocidade, ângulo, velocidade angular)
- **Recompensa**: +1 por passo

### **Pendulum-v1** (Contínuo)
- **Objetivo**: Controlar pêndulo para posição vertical
- **Ações**: Contínuas (-2 a +2 torque)
- **Estado**: 3 valores (cos, sin do ângulo, velocidade angular)
- **Recompensa**: Negativa (penaliza desequilíbrio)

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install gymnasium[classic_control] stable-baselines3
```

### Executar um algoritmo específico
```bash
python a2c_cartpole.py
python sac_pendulum.py
```

### Executar comparação completa
```bash
python comparacao_algoritmos.py
```

## 📊 Resultados Esperados

### CartPole (Objetivo: 500 pontos)
- **PPO/A2C**: Geralmente atingem 500 pontos
- **DQN**: Pode atingir 500 pontos com bom tuning

### Pendulum (Objetivo: ~0 pontos)
- **SAC/TD3**: Geralmente melhores resultados
- **DDPG**: Bom, mas pode ser instável

## 🔧 Personalização

### Modificar Hiperparâmetros
```python
# Exemplo: alterar learning rate
model = PPO("MlpPolicy", env, learning_rate=0.0003)
```

### Mudar Ambiente
```python
# Usar outro ambiente
env = make_vec_env("LunarLander-v2", n_envs=4)
```

### Ajustar Treinamento
```python
# Mais passos de treinamento
model.learn(total_timesteps=100000)
```

## 📈 Comparação de Performance

| Algoritmo | On/Off Policy | Espaço Ação | Estabilidade | Velocidade |
|-----------|---------------|-------------|--------------|------------|
| PPO       | On           | Discreto/Contínuo | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| A2C       | On           | Discreto/Contínuo | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| DQN       | Off          | Discreto    | ⭐⭐⭐ | ⭐⭐⭐ |
| SAC       | Off          | Contínuo    | ⭐⭐⭐⭐ | ⭐⭐ |
| TD3       | Off          | Contínuo    | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| DDPG      | Off          | Contínuo    | ⭐⭐⭐ | ⭐⭐ |

## 🎓 Conceitos Aprendidos

- **On-policy vs Off-policy**: Atualização da política
- **Value-based vs Policy-based**: Como aprender
- **Exploration vs Exploitation**: Equilíbrio na aprendizagem
- **Replay Buffer**: Reutilização de experiências
- **Target Networks**: Estabilidade no aprendizado

## 🔗 Recursos Adicionais

- [Stable Baselines 3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [Spinning Up RL](https://spinningup.openai.com/)
- [Reinforcement Learning Course](https://www.coursera.org/learn/reinforcement-learning)

---

**Dica**: Comece com PPO para problemas simples e SAC/TD3 para ambientes contínuos complexos!