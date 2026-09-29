# DrenNitaj_AI-Powered_Multi-Robot_Coordination_in_Dynamic_Environments
## Requirements

1. Implement a multi-agent reinforcement learning algorithm to train robots to cooperate.

2. Use an online path-planning algorithm (e.g., A* or RRT*) for dynamic obstacle avoidance.

3. Incorporate priority handling to resolve task conflicts dynamically.

4. Simulate the environment with changing conditions (e.g., moving obstacles, new tasks).

### Algorithms

- **Training:** Multi-Agent Deep Q-Learning (MADQL) or Proximal Policy Optimization (PPO).
- **Real-time Path Planning:** A* or RRT* with dynamic obstacle updates.
- **Task Allocation:** A market-based algorithm (e.g., auction-based assignment) based on priority and robot availability.

### Input

- Map of the environment, including dynamic obstacles.
- Initial robot positions and tasks (e.g., delivery locations).
- Reward structure for task completion and penalties for collisions or inefficiencies.

### Output

- Simulated execution of the robots achieving their tasks.

### Metrics

- Task completion time.
- Collisions avoided.
- Overall efficiency.

### Challenges

- Designing an RL reward system to balance efficiency, safety, and collaboration.
- Handling high-dimensional state spaces and dynamic changes in real-time.
- Ensuring scalability to larger teams and more complex environments.
