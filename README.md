<img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%">

<div align="center">

# 🕹️ AI Pac-Man Game
### Deep Convolutional Q-Learning for Pac-Man Mastery

<img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12-orange?style=for-the-badge&logo=pytorch)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-green?style=for-the-badge&logo=openaigym)

![download2-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/fb06c7f5-1474-41fb-938d-a96732d0e658)


</div>

<img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%">

## 🚀 Project Overview

This project implements an AI agent capable of mastering the classic Pac-Man game using Deep Convolutional Q-Learning. The agent learns to navigate the maze, eat pellets, and avoid ghosts by maximizing rewards through trial and error in a simulated environment.

<img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%">

## 🧠 Key Features

- **Deep Convolutional Q-Learning**: Utilizes advanced neural network architecture to process game states and learn optimal strategies.
- **Gymnasium Integration**: Leverages the Gymnasium library for a robust and flexible reinforcement learning environment.
- **Experience Replay**: Implements a memory buffer to store and sample past experiences, improving learning stability.
- **Target Network**: Uses a separate target network to reduce overestimation of Q-values and enhance training stability.
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation during the learning process.

<img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%">

## 💻 Tech Stack

<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="Python" width="50" height="50">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" alt="PyTorch" width="50" height="50">
    <img src="https://gymnasium.farama.org/_images/gymnasium-text.png" alt="Gymnasium" width="150" height="50">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" alt="NumPy" width="50" height="50">
<img src="https://img.icons8.com/?size=96&id=lOqoeP2Zy02f&format=png" alt="Colab" width="50" height="50">

<img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%">

## 🛠️ Implementation Highlights

1. **Neural Network Architecture**: 
```python
class Network(nn.Module):
  def __init__(self, action_size, seed = 42):
    super(Network, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1)
    self.bn4 = nn.BatchNorm2d(128)
    self.fc1 = nn.Linear(10 * 10 * 128, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, action_size)
```

2. **Agent Implementation**:
```python
class Agent():
  def __init__(self, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.action_size = action_size
    self.local_qnetwork = Network(action_size).to(self.device)
    self.target_qnetwork = Network(action_size).to(self.device)
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
    self.memory = deque(maxlen = 10000)
```

3. **Training Loop**:
```python
for episode in range(1, number_episodes + 1):
  state, _ = env.reset()
  score = 0
  for t in range(maximum_number_timesteps_per_episode):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  # ... (score tracking and epsilon decay)
```

<img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%">

## 📊 Performance

- **Training Episodes**: 2000
- **Average Score**: Improves over time, reaching 500+ in successful runs
- **Epsilon Decay**: From 1.0 to 0.01, ensuring a balance between exploration and exploitation
- **Learning Rate**: 5e-4
- **Discount Factor**: 0.99

<img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%">

## 🎮 How to Run

1. Clone the repository:
   ```
   git clone https://github.com/DeAtHfIrE26/AI_Pac_MAN_GAME.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```
   python train_pacman.py
   ```
4. Visualize results:
   ```
   python visualize_results.py
   ```

<img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%">

## 🚀 Future Enhancements

- Implement advanced algorithms like Double DQN or Dueling DQN
- Experiment with different neural network architectures
- Add support for multiple Atari games
- Integrate with MLflow for experiment tracking

<img src="https://raw.githubusercontent.com/SamirPaulb/SamirPaulb/main/assets/rainbow-superthin.webp" width="100%">

<div align="center">

Built with 💖 and 🧠 by Kashyap Patel

</div>
