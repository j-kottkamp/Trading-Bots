import numpy as np
import random
import datetime
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self.build_model()

    def build_model(self):
        # Neural Network for Deep Q-Learning
        model = tf.keras.Sequential([
            layers.Dense(64, input_dim=self.state_size, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array(state).reshape((1, -1))  # Adjust input shape
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state).reshape((1, -1))  # Adjust input shape
            next_state = np.array(next_state).reshape((1, -1))  # Adjust input shape
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
        self.total_profit = 0
        self.loss_penalty_multiplier = 1.1
        self.investment_percentage = 0.1

        self.compute_indicators()

    def compute_indicators(self):
        close_prices = self.price_data['Close'].values
        high_prices = self.price_data['High'].values
        low_prices = self.price_data['Low'].values
        volume = self.price_data['Volume'].values

        self.price_data['EMA_9'] = ta.EMA(close_prices, timeperiod=9)
        self.price_data['EMA_20'] = ta.EMA(close_prices, timeperiod=20)
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
        self.total_profit = 0
        return self.get_state()

    def get_state(self):
        current_price = self.price_data.iloc[self.current_step]
        state = [
            current_price['Close'],
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
            current_price['EMA_20'],
            current_price['EMA_50'],
            current_price['RSI'],
        ]
        return np.array(state)

    def step(self, action):
        current_price = self.price_data.iloc[self.current_step]
        price = current_price['Close']
        invest_amount = self.current_balance * self.investment_percentage

        if action == 1:  # Buy
            num_shares_to_buy = invest_amount // price
            if num_shares_to_buy > 0:
                self.shares_held += num_shares_to_buy
                self.current_balance -= num_shares_to_buy * price

        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.current_balance += self.shares_held * price
                self.shares_held = 0

        elif action == 3:  # Short (Sell short)
            num_shares_to_short = invest_amount // price
            if num_shares_to_short > 0:
                self.short_positions += num_shares_to_short
                self.current_balance += num_shares_to_short * price

        elif action == 4:  # Cover (Buy to cover short)
            if self.short_positions > 0:
                self.current_balance -= self.short_positions * price
                self.short_positions = 0

        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1

        next_price = self.price_data.iloc[self.current_step] if not done else current_price
        total_value = self.current_balance + self.shares_held * next_price['Close'] - self.short_positions * next_price['Close']
        profit = total_value - self.initial_investment

        if profit < 0:
            reward = profit * self.loss_penalty_multiplier
        else:
            reward = profit

        self.total_profit = profit
        return self.get_state(), reward, done

# Training the DQN Agent
def dqn_training(price_data, episodes=100, initial_investment=10000):
    env = TradingEnvironment(price_data, initial_investment)
    state_size = len(env.get_state())  # Adjust state size
    action_size = 5  # Buy, Sell, Hold, Short, Cover
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    episode_rewards = []
    
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
        
        episode_rewards.append(total_reward)
        
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward/100000:.2f}, Total Profit: {env.total_profit/100:.2f}, Epsilon: {agent.epsilon:.4f}")
    # Plot the rewards over episodes
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards Over Time')
    plt.show()

    agent.save("dqn_trading.h5")


# Download price data and run the training
SYMBOL = "SPY"
START = datetime.datetime(2021, 1, 1)
END = datetime.datetime(2025, 1, 1)
price_data = yf.download(SYMBOL, start=START, end=END)
dqn_training(price_data)
