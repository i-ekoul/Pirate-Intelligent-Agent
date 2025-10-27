# 🏴‍☠️ Pirate Treasure Hunt – Deep Q-Learning Intelligent Agent

**CS 370: Artificial Intelligence**

## 📘 Overview

This project implements an **intelligent agent** (a pirate) that navigates a maze to find treasure before the player does.
The pirate uses **Deep Q-Learning**, a reinforcement learning technique, to learn an optimal path through exploration and exploitation.

The system is trained to navigate an 8×8 maze, avoiding obstacles (walls) and finding the most efficient path to the treasure.

---

## 🎯 Project Objectives

This project demonstrates the following competencies:

* **Explain** the basic concepts and techniques of artificial intelligence and intelligent systems.
* **Analyze** how algorithms are used in AI to solve complex problems, particularly pathfinding with reinforcement learning.
* **Implement** a Deep Q-Learning algorithm to create a functional intelligent agent.
* **Use** industry-standard Python coding practices and maintain readable, well-documented code.

---

## 🧩 Maze Environment

The maze is represented as a NumPy 2D array where:

* `1.0` = Wall / Obstacle (impassable)
* `0.0` = Free space (traversable path)

Example maze used in training:

```python
maze = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  0.]  # Treasure location (goal)
])
```

* **Start position**: (0, 1)
* **Goal position**: (7, 7) – marked as the treasure
* **Walls**: Marked by 1’s
* **Valid paths**: Marked by 0’s

---

## 🧠 Algorithm Description – Deep Q-Learning

### 1. Environment

The pirate moves through a grid-based maze defined by the `TreasureMaze` class.
Each movement results in:

* A **reward** (positive for reaching treasure, negative for hitting a wall or losing)
* An updated **state** (the maze layout and pirate’s position)
* A **game status** (`win`, `lose`, or ongoing)

### 2. Agent (Neural Network Model)

A **Deep Neural Network** approximates the Q-function:

* **Input**: Flattened maze representation (the current state)
* **Output**: Q-values for each possible action (`LEFT`, `UP`, `RIGHT`, `DOWN`)
* **Architecture**:

  ```python
  Input → Dense(64, relu) → Dense(64, relu) → Output(4)
  ```
* **Loss Function**: Mean Squared Error (MSE)
* **Optimizer**: Adam

### 3. Experience Replay

The `GameExperience` class stores game episodes (`state, action, reward, next_state, status`).
During training, random batches of these experiences are replayed to improve learning stability.

### 4. Exploration vs. Exploitation

The **epsilon (ε)** parameter controls the balance:

* **Exploration** (random moves): Helps the pirate discover new paths.
* **Exploitation** (best known moves): Uses learned strategies to optimize the path.

Epsilon decays over time:

```python
epsilon = max(0.10, epsilon * 0.995)
```

### 5. Training

The model is trained using:

```python
qtrain(model, maze,
       n_epoch=200,
       max_memory=1000,
       data_size=8,
       max_steps=40)
```

* **n_epoch**: Number of training iterations
* **max_memory**: Size of replay buffer
* **data_size**: Batch size for training
* **max_steps**: Maximum steps per episode

During training, you’ll see outputs like:

```
Epoch: 015/199 | Loss: 0.0049 | Episodes: 141 | Win count: 9 | Win rate: 0.562 | time: 33.56 minutes
```

Training continues until the model achieves a 100% win rate or completes all epochs.

---

## 🏁 Results

* The pirate successfully learns to navigate from the top-left start position to the bottom-right treasure cell.
* Win rate improves steadily as the agent learns from experience.
* The final model consistently reaches the treasure within a few moves after sufficient training.

---

## 💻 How to Run

1. Launch the provided Jupyter Notebook:

   ```
   TreasureHuntGame.ipynb
   ```

2. Ensure dependencies are installed:

   ```bash
   pip install -r requirements.txt
   ```

3. Run all cells (`Cell → Run All`) to:

   * Build the model
   * Train the pirate agent
   * Visualize the final maze and movement sequence

4. Optionally, you can test the trained model from multiple start positions:

   ```python
   starts = [(0, 0), (3, 3), (7, 0)]
   for start in starts:
       play_game(model, qmaze, start)
       show(qmaze)
   ```

---

## 🧾 File Structure

```
📁 CS370_PirateTreasureHunt
├── TreasureHuntGame.ipynb          # Main Jupyter Notebook
├── TreasureMaze.py                 # Environment class (maze logic)
├── GameExperience.py               # Experience replay buffer
├── requirements.txt                # Python dependencies
├── Cole_Emmalie_ProjectTwo.html    # HTML export of notebook
└── README.md                       # Documentation (this file)
```

---

## 🧮 Technical Details

| Component                | Description                             |
| ------------------------ | --------------------------------------- |
| **Algorithm**            | Deep Q-Learning                         |
| **Neural Network**       | 2 Hidden Layers (64 neurons each, ReLU) |
| **Loss Function**        | Mean Squared Error (MSE)                |
| **Optimizer**            | Adam                                    |
| **Exploration Rate (ε)** | Starts at 0.2, decays to 0.05           |
| **Frameworks**           | TensorFlow / Keras, NumPy, Matplotlib   |

---

## 📈 Future Improvements

* Implement a **dynamic epsilon decay** based on performance stability.
* Visualize learning curves (loss and win rate over time).
* Add a **player vs. pirate mode** with simultaneous agents.
* Save and reload trained model weights for testing.

---

## 📚 References

* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction (2nd Edition).* MIT Press.
* François-Lavet, V., et al. (2018). *An Introduction to Deep Reinforcement Learning.* Foundations and Trends® in Machine Learning.
* TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)

---

## 🏆 Author

**Emmalie Cole**
CS 370 – Southern New Hampshire University
Project Two – *Pirate Intelligent Agent (Deep Q-Learning)*

---

Would you like me to format this README so it’s ready for **submission to Brightspace** (plain .docx with APA header and 12pt Times New Roman), or keep it as a clean Markdown (`README.md`) for GitHub or Codio?
