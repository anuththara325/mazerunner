import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from collections import deque

# Maze class
class OptimizedMaze:
    def __init__(self, maze, start_position, goal_position):
        self.maze = maze
        self.maze_height = maze.shape[0]
        self.maze_width = maze.shape[1]
        self.start_position = start_position
        self.goal_position = goal_position
        self.distance_map = self._compute_distance_map()
    
    def _compute_distance_map(self):
        distance_map = np.zeros((self.maze_height, self.maze_width))
        for i in range(self.maze_height):
            for j in range(self.maze_width):
                if self.maze[i][j] != 1:
                    distance_map[i][j] = abs(i - self.goal_position[1]) + abs(j - self.goal_position[0])
        return distance_map

# Q-learning agent
class ImprovedQLearningAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.95, 
                 exploration_start=1.0, exploration_end=0.05, num_episodes=1000):
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, 4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.target_q_table = self.q_table.copy()
        
    def get_action(self, state, current_episode):
        exploration_rate = self.exploration_end + (self.exploration_start - self.exploration_end) * \
                         np.exp(-3 * current_episode / self.num_episodes)
        
        if np.random.rand() < exploration_rate:
            return np.random.randint(4)
        
        q_values = self.q_table[state]
        return np.argmax(q_values)

# Reward function
def enhanced_reward_function(state, next_state, maze, done, steps):
    if done:
        return 500 / (steps + 1)
    if next_state == state:
        return -5
    current_distance = maze.distance_map[state[1]][state[0]]
    next_distance = maze.distance_map[next_state[1]][next_state[0]]
    distance_reward = 2 * (current_distance - next_distance)
    time_penalty = -0.05 * steps
    move_penalty = -0.1
    return distance_reward + time_penalty + move_penalty

# Maze layout and actions
maze_layout = np.array([
    [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
])

actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
start_position = (0, 0)
goal_position = (9, 9)

# Update the run_simulation function to include elapsed time correctly
def run_simulation(agent, maze, is_trained=False):
    current_state = start_position
    path = [current_state]
    total_reward = 0
    steps = 0
    rewards = []

    while steps < 100:  # Limit steps to prevent infinite loops
        if is_trained:
            state_q_values = agent.q_table[current_state[0], current_state[1]]
            action = np.argmax(state_q_values)
        else:
            action = agent.get_action(current_state, 0)  # High exploration for untrained
        
        next_state = (
            current_state[0] + actions[action][0],
            current_state[1] + actions[action][1]
        )
        
        if (next_state[0] < 0 or next_state[0] >= maze.maze_width or
            next_state[1] < 0 or next_state[1] >= maze.maze_height or
            maze.maze[next_state[1]][next_state[0]] == 1):
            next_state = current_state
        
        done = next_state == goal_position
        reward = enhanced_reward_function(current_state, next_state, maze, done, steps)
        rewards.append(reward)

        current_state = next_state
        path.append(current_state)
        total_reward += reward
        steps += 1
        
        if done:
            break
            
    return path, total_reward, steps, done, rewards

def main():
    st.title("Maze Solving: Trained vs Untrained Agent")
    
    maze = OptimizedMaze(maze_layout, start_position, goal_position)
    
    # Load trained agent
    try:
        with open('trained_maze_agent.pkl', 'rb') as f:
            saved_agent = pickle.load(f)
        trained_agent = ImprovedQLearningAgent(maze)
        trained_agent.q_table = saved_agent['q_table']
        trained_agent.target_q_table = saved_agent['target_q_table']
        st.success("Trained agent loaded successfully!")
    except Exception as e:
        st.error(f"Could not load trained agent. Ensure 'trained_maze_agent.pkl' exists. Error: {e}")
        return
    
    untrained_agent = ImprovedQLearningAgent(maze)
    st.sidebar.header("Simulation Controls")
    simulation_speed = st.sidebar.slider("Simulation Speed", 0.1, 2.0, 0.5)
    start_button = st.sidebar.button("Start Simulation")
    
    if start_button:
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Untrained Agent")
            untrained_metrics = st.empty()
            untrained_plot = st.empty()
        
        with col2:
            st.header("Trained Agent")
            trained_metrics = st.empty()
            trained_plot = st.empty()
        
        # Start the timer here
        start_time = time.time()
        
        # Create figures for both agents
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        
        untrained_path, untrained_reward, untrained_steps, untrained_done, untrained_rewards = run_simulation(untrained_agent, maze, False)
        trained_path, trained_reward, trained_steps, trained_done, trained_rewards = run_simulation(trained_agent, maze, True)
        
        max_steps = max(len(untrained_path), len(trained_path))
        
        for step in range(max_steps):
            elapsed_time = time.time() - start_time  # Update elapsed time here
            
            if step < len(untrained_path):
                ax1.clear()
                ax1.imshow(maze_layout, cmap='gray')
                path_array = np.array(untrained_path[:step+1])
                ax1.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, alpha=0.7)
                ax1.plot(path_array[:, 0], path_array[:, 1], 'b.', markersize=10)
                ax1.text(start_position[0], start_position[1], 'S', ha='center', va='center', color='green', fontsize=20)
                ax1.text(goal_position[0], goal_position[1], 'G', ha='center', va='center', color='red', fontsize=20)
                ax1.set_title(f'Untrained Agent\nStep: {step+1}')
                untrained_plot.pyplot(fig1)
                
                untrained_metrics.markdown(f"""
                - Steps: {step+1}
                - Cumulative Reward: {sum(untrained_rewards[:step+1]):.2f}
                - Time Taken: {elapsed_time:.2f}s
                """)
            
            if step < len(trained_path):
                ax2.clear()
                ax2.imshow(maze_layout, cmap='gray')
                path_array = np.array(trained_path[:step+1])
                ax2.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=2, alpha=0.7)
                ax2.plot(path_array[:, 0], path_array[:, 1], 'r.', markersize=10)
                ax2.text(start_position[0], start_position[1], 'S', ha='center', va='center', color='green', fontsize=20)
                ax2.text(goal_position[0], goal_position[1], 'G', ha='center', va='center', color='red', fontsize=20)
                ax2.set_title(f'Trained Agent\nStep: {step+1}')
                trained_plot.pyplot(fig2)
                
                trained_metrics.markdown(f"""
                - Steps: {step+1}
                - Cumulative Reward: {sum(trained_rewards[:step+1]):.2f}
                - Time Taken: {elapsed_time:.2f}s
                """)
            
            time.sleep(1 / simulation_speed)
        
        st.markdown("### Final Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Untrained Agent**")
            st.markdown(f"- Total Steps: {untrained_steps}")
            st.markdown(f"- Total Reward: {untrained_reward:.2f}")
            st.markdown(f"- Goal Reached: {'Yes' if untrained_done else 'No'}")
        with col2:
            st.markdown("**Trained Agent**")
            st.markdown(f"- Total Steps: {trained_steps}")
            st.markdown(f"- Total Reward: {trained_reward:.2f}")
            st.markdown(f"- Goal Reached: {'Yes' if trained_done else 'No'}")

if __name__ == "__main__":
    main()
