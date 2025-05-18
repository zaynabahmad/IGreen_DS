import random 
import gym 
import numpy as np 

env = gym.make('Taxi-v3')

alpha = 0.9     
gamma = 0.95           
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon  = 0.01
num_episodes = 10000
max_steps = 100 

q_table = np.zeros((env.observation_space.n , env.action_space.n))

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

for episode in range(num_episodes): 
    state, _ = env.reset() 
    done = False

    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state
        if done or truncated: 
            break 

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Test the trained agent
test_env = gym.make('Taxi-v3', render_mode='human')

for episode in range(5): 
    state, _ = test_env.reset()
    done = False
    print('Episode', episode)

    for step in range(max_steps): 
        test_env.render()
        action = np.argmax(q_table[state])
        next_state, reward, done, truncated, info = test_env.step(action)
        state = next_state

        if done or truncated: 
            test_env.render() 
            print('Finished episode', episode, 'with reward', reward)
            break 

test_env.close()
