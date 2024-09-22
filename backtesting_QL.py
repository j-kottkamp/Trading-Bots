import numpy as np
import random
import datetime
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
import optuna
from scipy.ndimage import median_filter


# Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  # Adjust to the number of actions (Buy, Sell, Short, Cover)
        self.q_table = {}  # Use a dictionary for Q-table to handle hashed states
        self.alpha = 0.5  # Learning rate
        self.gamma = 0.85  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate (reduced over time)
        self.epsilon_decay = 0.99 # Decay rate for exploration
        self.epsilon_min = 0.01  # Minimum value of epsilon

    def choose_action(self, state):
        state_key = self.hash_state(state)  # Convert state to a hashable key
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  # Adjust for the new number of actions
        if state_key not in self.q_table:
            return random.choice(range(self.action_size))  # If state is not in Q-table, choose random action
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state):
        state_key = self.hash_state(state)
        next_state_key = self.hash_state(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        best_next_action = np.argmax(self.q_table[next_state_key])
        target = reward + self.gamma * self.q_table[next_state_key][best_next_action]
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action])

    def hash_state(self, state):
        return tuple(state)


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
        self.investment_percentage = 0.1
        self.previous_total_value = initial_investment
        self.returns = []
        self.risk_free_rate = 0.01  # Assuming a small risk-free rate
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
        self.returns = []
        return self.get_state()

    def get_state(self):
        current_price = self.price_data.iloc[self.current_step]
        state = {
            'current_price': current_price['Close'],
            'ATR': current_price['ATR'],
            'MACD': current_price['MACD'],
            'MACD_signal': current_price['MACD_signal'],
            'MACD_hist': current_price['MACD_hist'],
            'PSAR': current_price['PSAR'],
            'ADX': current_price['ADX'],
            'VAR': current_price['VAR'],
            'TSF': current_price['TSF'],
            'HT_DCPHASE': current_price['HT_DCPHASE'],
            'HT_DCPeriod': current_price['HT_DCPeriod'],
            'EMA_9': current_price['EMA_9'],
            'EMA_20': current_price['EMA_20'],
            'EMA_50': current_price['EMA_50'],
            'RSI': current_price['RSI'],
        }
        return state

    def step(self, action):
        current_price = self.price_data.iloc[self.current_step]['Close']
        invest_amount = self.current_balance * self.investment_percentage
        reward = 0

        # Buy action
        if action == 1:  # Buy
            num_shares_to_buy = invest_amount // current_price
            if num_shares_to_buy > 0:
                self.shares_held += num_shares_to_buy
                self.current_balance -= num_shares_to_buy * current_price

        # Sell action
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.current_balance += self.shares_held * current_price
                self.shares_held = 0

        # Short action
        elif action == 3:  # Short (Sell short)
            num_shares_to_short = invest_amount // current_price
            if num_shares_to_short > 0:
                self.short_positions += num_shares_to_short
                self.current_balance += num_shares_to_short * current_price

        # Cover short action
        elif action == 4:  # Cover (Buy to cover short)
            if self.short_positions > 0:
                self.current_balance -= self.short_positions * current_price
                self.short_positions = 0

        self.current_step += 1

        # Calculate portfolio value
        current_total_value = self.current_balance + self.shares_held * current_price - self.short_positions * current_price
        profit = current_total_value - self.previous_total_value

        # Track returns
        if self.previous_total_value > 0:
            step_return = profit / self.previous_total_value
            self.returns.append(step_return)

        # Reward as percentage change in portfolio value
        if self.previous_total_value > 0:
            reward = ((current_total_value - self.previous_total_value) / self.previous_total_value) * 1000

        # Update total profit and value
        self.previous_total_value = current_total_value
        self.total_profit += current_total_value - self.initial_investment

        # End of data?
        done = self.current_step >= len(self.price_data) - 1

        return self.get_state(), reward, done


# Test and train the Q-Learning agent
def reinforcement_learning(price_data, episodes=1000, initial_investment=10000):
    env = TradingEnvironment(price_data, initial_investment)
    agent = QLearningAgent(state_size=len(price_data.columns), action_size=5)  # Action size auf 5 erhöht
    
    episode_rewards = []
    episode_profits = []

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_profits.append(env.total_profit)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward/10:.2f}, Total Profit: {env.total_profit:.2f}, Epsilon: {agent.epsilon:.4f}")


    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # Beispiel: Anwenden eines gleitenden Durchschnitts
    window_size = 20
    episode_rewards_smoothed = moving_average(np.array(episode_rewards), window_size)
            
    # Lernkurven plotten: Belohnungen vor und nach Sharpe-Adjustment
    def plot_rewards_profits_with_median_filter(episode_rewards, episode_profits, filter_size=5):
        # Apply the median filter to rewards and profits
        filtered_rewards = median_filter(episode_rewards, size=filter_size)
        filtered_profits = median_filter(episode_profits, size=filter_size)
        
        # Plot original rewards and filtered rewards on different charts
        plt.figure(figsize=(14, 10))

        # Original Rewards
        plt.subplot(2, 2, 1)
        plt.plot(episode_rewards, label='Original Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Original Reward per Episode')
        plt.legend()

        # Filtered Rewards
        plt.subplot(2, 2, 2)
        plt.plot(filtered_rewards, label='Filtered Reward', color='green')
        plt.xlabel('Episode')
        plt.ylabel('Filtered Reward')
        plt.title('Filtered Reward per Episode (Median Filter)')
        plt.legend()

        # Original Profits
        plt.subplot(2, 2, 3)
        plt.plot(episode_profits, label='Original Profit', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Total Profit')
        plt.title('Original Profit per Episode')
        plt.legend()

        # Filtered Profits
        plt.subplot(2, 2, 4)
        plt.plot(filtered_profits, label='Filtered Profit', color='purple')
        plt.xlabel('Episode')
        plt.ylabel('Filtered Profit')
        plt.title('Filtered Profit per Episode (Median Filter)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Example call to the function (assuming episode_rewards and episode_profits are defined)
    plot_rewards_profits_with_median_filter(episode_rewards, episode_profits, filter_size=5)
    

    return agent

# Download price data and run the training
SYMBOL = "SPY"
START = datetime.datetime(2021, 1, 1)
END = datetime.datetime(2025, 1, 1)
price_data = yf.download(SYMBOL, start=START, end=END)
agent = reinforcement_learning(price_data)


def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-4, 1e-1)
    gamma = trial.suggest_float('gamma', 0.1, 0.99)
    epsilon = trial.suggest_float('epsilon', 0.1, 1.0)
    
    # Initialize environment and agent with current hyperparameters
    env = TradingEnvironment(price_data)
    agent = QLearningAgent(state_size=len(price_data.columns), action_size=3)
    agent.alpha = alpha
    agent.gamma = gamma
    agent.epsilon = epsilon
    
    episode_rewards = []
    
    for e in range(500):  # Number of episodes for evaluation
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
    
    return np.mean(episode_rewards)

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# print("Best parameters found: ", study.best_params)
