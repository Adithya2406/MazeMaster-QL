# MazeMaster-QL - 3D Maze Navigation using Q(λ) Reinforcement Learning
A reinforcement learning agent using Q(λ) with eligibility traces to master 60x60 mazes, featuring an automated training pipeline and detailed 3D visualizations.
This project was developed as part of CS 271P – Intro to Artificial Intelligence (Fall 2025, UC Irvine).

🚀 Project Overview
This repository implements an agent that learns to navigate complex mazes without prior knowledge of the environment. The agent uses the Q(λ) algorithm with eligibility traces, enabling rapid convergence on long-horizon tasks where rewards are sparse.

The system includes:
Procedurally generated mazes (5×5 to 50×50)
Q(λ) reinforcement learning agent
BFS baseline comparison

Full visualization suite:
3D maze rendering
Top-down animation of agent trajectory
Third-person follow camera
First-person "robot-eye" view
Model saving, evaluation scripts, and performance metrics

🧠 Key Features
✔️ Q(λ) Learning with Eligibility Traces - Allows the agent to learn long paths efficiently by assigning credit backward along visited states.

✔️ 3D Visualizations
exploration3d.gif → 3D maze + top-down rollout
exploration3d_fp.gif → First-person pseudo-3D view
exploration3d_3rd.gif → Third-person robot follow camera
maze_3d.png → Static 3D map
rl_solution.png, value_heatmap.png, visitation_heatmap.png

✔️ Fully Modular Codebase
maze_generator.py → Randomized recursive maze
qlambda_agent.py → Q(λ) algorithm implementation
maze_env_3d_wrapper.py → Environment logic
visualize_3d.py → 3D rendering tools
train.py → Full training loop
evaluate.py → Batch evaluation using multiprocessing

✔️ Strong Baseline Evaluation
BFS for optimal shortest path
Success-rate, steps, reward curves
Path optimality ratio (RL / BFS)

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

🏃 Training the Agent
Run: python train.py --size 50 --episodes 2000 --difficulty medium
Common arguments:
--size N             Maze size (NxN)
--episodes K         Number of training episodes
--difficulty {easy,medium,hard}
--max_steps M        Max steps per episode
Outputs (GIFs + plots) will be saved in outputs/.

📊 Evaluation
After training: python evaluate.py --episodes 200 --procs 8
This script loads:
models/maze.npy
models/qtable.npy

And outputs:
Success rate
Average path-length ratio (RL vs BFS)

🎥 Generated Visualizations
Example outputs:
🔹 3D Maze Exploration (Top-down + 3D)
outputs/exploration3d.gif
🔹 First-Person Robot POV
outputs/exploration3d_fp.gif
🔹 Third-Person Follow Camera
outputs/exploration3d_3rd.gif
🔹 Learning Plots
Value heatmap
Visitation heatmap
RL solution path
Training curves

📈 Performance Metrics
After 300–400 episodes, the agent reaches:
Metric	Value
Success Rate	100%
Greedy Path Length	Equal to BFS
Eval Success (200 runs)	100%
Path Optimality Ratio	~1.00

🧩 Challenges & Solutions
Challenge	How It Was Addressed
Sparse rewards	Eligibility traces + step penalty
Long exploration loops	ε-decay & Q(λ) propagation
Rendering lag	Cached cube geometry
Camera stabilization	Smoothed azimuth changes
3D visibility	Semi-transparent walls + higher path contrast

📚 References
Sutton & Barto – Reinforcement Learning: An Introduction
Watkins & Dayan – Q-learning paper
Matplotlib 3D engine
Maze generation algorithms (recursive backtracking)

🙌 Acknowledgements
This project was developed for CS 271P, under instruction from Kalev Kask.
