import numpy as np
import random
import datetime
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
from keras.layers import Dense, LSTM
import time
import cProfile
import pstats

# DQN Agent
class PrioritizedReplayBuffer:
    def __init__(self, size, alpha):
        self.size = size
        self.alpha = alpha
        self.buffer = deque(maxlen=size)
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.pos = 0

    def add(self, experience, priority):
        if len(self.buffer) < self.size:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.size

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], []

        # Use the size of the buffer if it is smaller than batch_size
        size = min(len(self.buffer), batch_size)

        priorities = self.priorities[:len(self.buffer)] + 1e-5  # Small constant to avoid zero priorities
        probabilities = priorities / np.sum(priorities)
        
        if np.isnan(probabilities).any():
            raise ValueError("Probabilities contain NaN")

        indices = np.random.choice(len(self.buffer), size, p=probabilities[:len(self.buffer)])
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, indices, priorities):
        valid_priorities = np.maximum(priorities, 1e-5)  # Avoid zero priorities
        self.priorities[indices] = valid_priorities ** self.alpha


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(size=2000, alpha=0.6)  # Updated buffer size
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9925
        self.learning_rate = 0.01
        self.batch_size = 64  # Updated batch size
        self.model = self.build_model()

    def remember(self, state, action, reward, next_state, done):
        priority = abs(reward)  # Calculate priority based on reward
        self.memory.add((state, action, reward, next_state, done), priority)

    def build_model(self):
        # Neural Network for Deep Q-Learning
        model = tf.keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array(state).reshape((1, -1))  # Adjust input shape
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        
        minibatch, indices = self.memory.sample(self.batch_size)
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        targets = rewards + self.gamma * (1 - dones) * np.amax(self.model.predict(next_states, verbose=0), axis=1)
        target_f = self.model.predict(states, verbose=0)
        target_f[np.arange(self.batch_size), actions] = targets
        
        self.model.fit(states, target_f, epochs=1, verbose=0)
        errors = np.abs(targets - target_f[np.arange(self.batch_size), actions])
        self.memory.update_priorities(indices, errors)



    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# Trading Environment
class TradingEnvironment:
    def __init__(self, price_data, initial_investment=10000):
        self.price_data = price_data
        self.current_step = 0
        self.initial_investment = initial_investment
        self.current_balance = initial_investment
        self.shares_held = 0
        self.short_positions = 0
        self.previous_total_value = initial_investment
        self.current_position = None
        self.returns = []
        self.investment_percentage = 0.1
        self.total_profit = 0

        # Additional variables for rewards
        self.previous_trade_value = 0
        self.trade_start_value = 0

        self.compute_indicators()

    def compute_indicators(self):
        close_prices = self.price_data['Close'].values
        high_prices = self.price_data['High'].values
        low_prices = self.price_data['Low'].values
        volume = self.price_data['Volume'].values

        self.price_data['EMA_9'] = ta.EMA(close_prices, timeperiod=9)
        self.price_data['EMA_50'] = ta.EMA(close_prices, timeperiod=50)
        self.price_data['RSI'] = ta.RSI(close_prices, timeperiod=14)
        self.price_data['ATR'] = ta.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        self.price_data['MACD'], self.price_data['MACD_signal'], self.price_data['MACD_hist'] = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        self.price_data['PSAR'] = ta.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2)
        self.price_data['ADX'] = ta.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        self.price_data['VAR'] = ta.VAR(close_prices, timeperiod=5)
        self.price_data['TSF'] = ta.TSF(close_prices, timeperiod=14)
        self.price_data['HT_DCPHASE'] = ta.HT_DCPHASE(close_prices)
        self.price_data['HT_DCPeriod'] = ta.HT_DCPERIOD(close_prices)
        
    def reset(self):
        self.current_step = 0
        self.current_balance = self.initial_investment
        self.shares_held = 0
        self.short_positions = 0
        self.previous_total_value = self.initial_investment
        self.current_position = None
        self.total_profit = 0  # Reset total profit
        return self.get_state()

    def get_state(self):
        current_price = self.price_data.iloc[self.current_step]
        state = [
            current_price['Close'],
            current_price['High'],
            current_price['Low'],
            current_price['Volume'],
            current_price['ATR'],
            current_price['MACD'],
            current_price['MACD_signal'],
            current_price['MACD_hist'],
            current_price['PSAR'],
            current_price['ADX'],
            current_price['VAR'],
            current_price['TSF'],
            current_price['HT_DCPHASE'],
            current_price['HT_DCPeriod'],
            current_price['EMA_9'],
            current_price['EMA_50'],
            current_price['RSI'],
        ]
        state_array = np.array(state)
        return state_array

    def step(self, action):
        current_price = self.price_data.iloc[self.current_step]
        price = current_price['Close']
        invest_amount = self.current_balance * self.investment_percentage
        reward = 0
        if action == 0:
            reward == -5
        
        if action == 1:  # Buy
            num_shares_to_buy = invest_amount // price
            if num_shares_to_buy > 0:
                self.shares_held += num_shares_to_buy
                self.current_balance -= num_shares_to_buy * price
                self.trade_start_value = price  # Track entry price
                reward = 10
                self.current_position = {
                    'type': 'buy',
                    'price': price,
                    'amount': num_shares_to_buy
                }

        elif action == 2:  # Sell
            if self.shares_held > 0 and self.current_position and self.current_position['type'] == 'buy':
                profit = (price - self.current_position['price']) * self.current_position['amount']
                self.total_profit += profit
                reward = self.calculate_reward(profit)
                
                self.current_balance += self.shares_held * price
                self.shares_held = 0
                self.current_position = None

        elif action == 3:  # Short
            num_shares_to_short = invest_amount // price
            if num_shares_to_short > 0:
                self.short_positions += num_shares_to_short
                self.current_balance += num_shares_to_short * price
                self.trade_start_value = price  # Track entry price for shorts
                reward = 10
                self.current_position = {
                    'type': 'short',
                    'price': price,
                    'amount': num_shares_to_short
                }

        elif action == 4:  # Cover
            if self.short_positions > 0 and self.current_position and self.current_position['type'] == 'short':
                profit = (self.current_position['price'] - price) * self.current_position['amount']
                self.total_profit += profit
                reward = self.calculate_reward(profit)
                
                self.current_balance -= self.short_positions * price
                self.short_positions = 0
                self.current_position = None
                

        self.current_step += 1
        # Calculate portfolio value
        current_total_value = self.current_balance + self.shares_held * price - self.short_positions * price
        profit = current_total_value - self.previous_total_value

        # Track returns
        if self.previous_total_value > 0:
            step_return = profit / self.previous_total_value
            self.returns.append(step_return)

        # Update previous total value
        self.previous_total_value = current_total_value

        # End of data?
        done = self.current_step >= len(self.price_data) - 1

        return self.get_state(), reward, done

    def calculate_reward(self, profit):
        if profit > 0:
            reward = (profit / self.previous_trade_value) * 1000 if self.previous_trade_value != 0 else 0
            if profit / self.previous_trade_value >= 0.05:
                reward += 4000  # Bonus for >5% profit
            elif profit / self.previous_trade_value >= 0.03:
                reward += 2000  # Bonus for >3% profit
            elif profit / self.previous_trade_value >= 0.01:
                reward += 500  # Bonus for >1% profit
        else:
            reward = (profit / self.previous_trade_value) * 1200 if self.previous_trade_value != 0 else 0  # Loss penalty

        self.previous_trade_value = abs(profit) if profit != 0 else self.previous_trade_value
        return reward



def preprocess_data(price_data):
    # Extract features without labels
    features = price_data[['Close', 'High', 'Low', 'Volume', 'EMA_9', 'EMA_50', 'RSI',
                           'ATR', 'MACD_hist', 'PSAR', 'ADX', 'VAR', 'TSF', 'HT_DCPHASE', 'HT_DCPeriod']].values
    
    # Create the dataset without labels
    dataset = tf.data.Dataset.from_tensor_slices(features)
    return dataset

# Training the DQN Agent
def dqn_training(price_data, episodes=500, initial_investment=10000):
    env = TradingEnvironment(price_data, initial_investment)
    state_size = len(env.get_state())
    action_size = 5  # Buy, Sell, Hold, Short, Cover
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Preprocess the data and create the dataset
    dataset = preprocess_data(price_data)
    BATCH_SIZE = 64
    dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    episode_rewards = []
    start_time = time.time()

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        for batch_features in dataset:  # Iterate through the dataset
            current_state = env.get_state()
            action = agent.choose_action(current_state)
            next_state, reward, done = env.step(action)
            agent.remember(current_state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break

        agent.replay()  # Replay after collecting a full episode
        
        # Decay epsilon at the end of each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        episode_rewards.append(total_reward)
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Total Profit: {env.total_profit:.2f}, Epsilon: {agent.epsilon:.4f}")

    end_time = time.time()
    
    # Calculate and print the total time taken
    total_time = end_time - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    # Plot the rewards over episodes
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards Over Time')
    plt.show()

    agent.save("D:/Python Codes/Python/DQN_Stuff/dqn_trading.weights.h5")






# Download price data and run the training
def main():
    SYMBOL = "SPY"
    START = datetime.datetime(2021, 1, 1)
    END = datetime.datetime(2025, 1, 1)
    price_data = yf.download(SYMBOL, start=START, end=END)
    dqn_training(price_data)

# Profile the main function
cProfile.run('main()', 'profile_output')

# View the results in a human-readable form
with open('D:/Python Codes/Python/DQN_Stuff/bottleNeckDQL.txt', 'w') as f:
    p = pstats.Stats('profile_output', stream=f)
    p.strip_dirs().sort_stats('cumtime').print_stats(20)  # Sort by cumulative time, print top 20
