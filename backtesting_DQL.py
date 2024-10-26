import numpy as np
import random
import datetime
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import time
import cProfile
import pstats

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Is cuDNN available: ", tf.test.is_built_with_cuda())

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
        self.memory = PrioritizedReplayBuffer(size=1000, alpha=0.6)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.98
        self.learning_rate = 0.05
        self.batch_size = 64
        self.model = self.build_model()

    def remember(self, state, action, reward, next_state, done):
        priority = abs(reward)
        self.memory.add((state, action, reward, next_state, done), priority)

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=(self.state_size,)),
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
        state = np.array(state).reshape((1, -1))
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
        self.previous_trade_value = 0
        self.days_in_trade = 0
        self.preprocessed_data = self.compute_indicators()


    def compute_indicators(self):
        close_prices = self.price_data['Close'].values
        high_prices = self.price_data['High'].values
        low_prices = self.price_data['Low'].values
        volume = self.price_data['Volume'].values

        indicators = np.array([
            close_prices,
            high_prices,
            low_prices,
            volume,
            ta.EMA(close_prices, timeperiod=9),
            ta.EMA(close_prices, timeperiod=50),
            ta.RSI(close_prices, timeperiod=14),
            ta.ATR(high_prices, low_prices, close_prices, timeperiod=14),
            ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)[2],  # MACD histogram
            ta.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2),
            ta.ADX(high_prices, low_prices, close_prices, timeperiod=14),
            ta.VAR(close_prices, timeperiod=5),
            ta.TSF(close_prices, timeperiod=14),
            ta.HT_DCPHASE(close_prices),
            ta.HT_DCPERIOD(close_prices)
        ]).T
        return indicators

    def reset(self):
        self.current_step = 0
        self.current_balance = self.initial_investment
        self.shares_held = 0
        self.short_positions = 0
        self.previous_total_value = self.initial_investment
        self.current_position = None
        self.total_profit = 0
        return self.get_state()

    def get_state(self):
        return self.preprocessed_data[self.current_step]

    def step(self, action):
        # Current price of the asset
        current_price = self.preprocessed_data[self.current_step][0]

        # Calculate the amount to invest based on current balance and investment percentage
        invest_amount = self.current_balance * self.investment_percentage
        reward = 0
        
        def current_total_value():
            self.current_total_value = self.current_balance + (self.shares_held * current_price) - (self.short_positions * current_price)
            return self.current_total_value

        # Actions: Hold, Buy, Sell, Short, Cover
        if action == 0:  # Hold
            reward = -5
            if self.current_position:  # If in a position, increment days in trade
                self.days_in_trade += 1

        elif action == 1:  # Buy
            if self.current_position:  # If in a position, increment days in trade
                self.days_in_trade += 1
            num_shares_to_buy = invest_amount // current_price
            if num_shares_to_buy > 0:
                self.shares_held += num_shares_to_buy
                self.current_balance -= num_shares_to_buy * current_price
                reward = 10
                self.current_position = {
                    'type': 'buy',
                    'price': current_price,
                    'amount': num_shares_to_buy
                }
                self.days_in_trade = 0

        elif action == 2:  # Sell
            if self.current_position:  # If in a position, increment days in trade
                self.days_in_trade += 1
            if self.shares_held > 0 and self.current_position and self.current_position['type'] == 'buy':
                profit = (current_price - self.current_position['price']) * self.current_position['amount']
                self.total_profit += profit
                self.current_balance += self.shares_held * current_price
                self.shares_held = 0
                
                reward = self.calculate_reward(profit)  # Pass current_total_value to reward calculation
                self.current_position = None
                self.days_in_trade = 0

        elif action == 3:  # Short
            if self.current_position:  # If in a position, increment days in trade
                self.days_in_trade += 1
            num_shares_to_short = invest_amount // current_price
            if num_shares_to_short > 0:
                self.short_positions += num_shares_to_short
                self.current_balance += num_shares_to_short * current_price
                reward = 10
                self.current_position = {
                    'type': 'short',
                    'price': current_price,
                    'amount': num_shares_to_short
                }
                self.days_in_trade = 0

        elif action == 4:  # Cover
            if self.current_position:  # If in a position, increment days in trade
                self.days_in_trade += 1
            if self.short_positions > 0 and self.current_position and self.current_position['type'] == 'short':
                profit = (self.current_position['price'] - current_price) * self.current_position['amount']
                self.total_profit += profit
                self.current_balance -= self.short_positions * current_price
                self.short_positions = 0

                reward = self.calculate_reward(profit)  # Pass current_total_value to reward calculation
                self.current_position = None
                self.days_in_trade = 0

        

        # Update step count
        self.current_step += 1

        # Store current total value and previous total value
        self.previous_total_value = current_total_value()

        # Check if the episode is done
        done = self.current_step >= len(self.preprocessed_data) - 1

        return self.get_state(), reward, done

    def calculate_reward(self, profit):
        if profit > 0:
            reward = (profit * 1000) / (self.days_in_trade * 0.01)
        else:
            reward = (profit * 1050)
        return reward




def preprocess_data(price_data):
    close_prices = price_data['Close'].values
    high_prices = price_data['High'].values
    low_prices = price_data['Low'].values
    volume = price_data['Volume'].values

    indicators = np.array([
        close_prices,
        high_prices,
        low_prices,
        volume,
        ta.EMA(close_prices, timeperiod=9),
        ta.EMA(close_prices, timeperiod=50),
        ta.RSI(close_prices, timeperiod=14),
        ta.ATR(high_prices, low_prices, close_prices, timeperiod=14),
        ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)[2],
        ta.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2),
        ta.ADX(high_prices, low_prices, close_prices, timeperiod=14),
        ta.VAR(close_prices, timeperiod=5),
        ta.TSF(close_prices, timeperiod=14),
        ta.HT_DCPHASE(close_prices),
        ta.HT_DCPERIOD(close_prices)
    ]).T
    return indicators

def initialize_environment(price_data, initial_investment=10000):
    """Initialize the trading environment."""
    env = TradingEnvironment(price_data, initial_investment)
    env.preprocessed_data = preprocess_data(price_data)
    return env

def initialize_agent(env):
    """Initialize the DQN agent based on the environment's state size."""
    state_size = len(env.get_state())
    action_size = 5  # Buy, Sell, Hold, Short, Cover
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    return agent

def calculate_drawdown(current_value, peak_value):
    """Calculate the drawdown given current and peak portfolio values."""
    drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
    return drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """Calculate the Sharpe ratio based on portfolio returns."""
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return (mean_return - risk_free_rate) / std_return if std_return != 0 else 0

def calculate_trade_metrics(reward, current_value, trade_action, episode_trade_gains):
    """Track trade-related metrics."""
    if trade_action in [1, 2, 3, 4]:  # Only count for actual trades
        trade_gain_percent = (reward / current_value) * 100
        episode_trade_gains.append(trade_gain_percent)
    return episode_trade_gains

def process_episode(env, agent, initial_investment):
    """Run a single training episode and return metrics."""
    state = env.reset()
    done = False
    total_reward, trade_count, wins, losses = 0, 0, 0, 0
    episode_returns, episode_trade_gains = [], []
    max_drawdown = 0
    peak_value = initial_investment
    
    while not done:
        current_state = env.get_state()
        action = agent.choose_action(current_state)
        next_state, reward, done = env.step(action)
        agent.remember(current_state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if action in [1, 2, 3, 4]:
            trade_count += 1

        # Track wins/losses and returns for Sharpe ratio
        if reward > 0: wins += 1
        elif reward < 0: losses += 1
        episode_returns.append(reward)

        # Track trade gain and drawdown
        episode_trade_gains = calculate_trade_metrics(reward, env.current_total_value, action, episode_trade_gains)
        current_value = env.current_total_value
        peak_value = max(peak_value, current_value)
        max_drawdown = max(max_drawdown, calculate_drawdown(current_value, peak_value))

    return total_reward, trade_count, wins, losses, episode_returns, episode_trade_gains, max_drawdown

def log_metrics(env, initial_investment, trade_count, wins, losses, episode_returns, episode_trade_gains):
    """Log and calculate metrics such as win rate, Sharpe ratio, and profit."""
    # Win rate
    win_rate = wins / trade_count if trade_count > 0 else 0

    # Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(episode_returns)

    # Average percentage gain per trade
    avg_percent_gain = np.mean(episode_trade_gains) if episode_trade_gains else 0

    # Total profit
    current_price = env.preprocessed_data[env.current_step][0]
    total_profit = env.current_balance + (env.shares_held * current_price) - (env.short_positions * current_price) - initial_investment

    return win_rate, sharpe_ratio, avg_percent_gain, total_profit

def plot_metrics(episode_rewards, num_trades, win_rates, sharpe_ratios, avg_percent_gains, max_drawdowns, total_profits):
    """Plot metrics over all episodes."""
    plt.figure(figsize=(12, 10))

    metrics = [
        (episode_rewards, 'Total Reward', 'Episode Rewards Over Time'),
        (num_trades, 'Number of Trades', 'Number of Trades Per Episode'),
        (sharpe_ratios, 'Sharpe Ratio', 'Sharpe Ratio Per Episode'),
        (total_profits, 'Total Profit', 'Total Profit Per Episode'),
        (win_rates, 'Win Rate', 'Win Rate Per Episode'),
        (avg_percent_gains, 'Avg % Gain/Trade', 'Average % Gain Per Trade'),
        (max_drawdowns, 'Max Drawdown', 'Max Drawdown Per Episode'),
        
        
        
    ]

    for idx, (data, ylabel, title) in enumerate(metrics):
        plt.subplot(3, 3, idx + 1)
        plt.plot(data)
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.title(title)

    plt.tight_layout()
    plt.show()

def dqn_training(price_data, episodes=10, initial_investment=10000):
    """Main function for DQN training over multiple episodes."""
    env = initialize_environment(price_data, initial_investment)
    agent = initialize_agent(env)

    # Metrics tracking
    episode_rewards, num_trades, win_rates, sharpe_ratios, avg_percent_gains, max_drawdowns, total_profits = [], [], [], [], [], [], []
    
    start_time = time.time()
    
    for e in range(episodes):
        total_reward, trade_count, wins, losses, episode_returns, episode_trade_gains, max_drawdown = process_episode(env, agent, initial_investment)
        
        win_rate, sharpe_ratio, avg_percent_gain, total_profit = log_metrics(env, initial_investment, trade_count, wins, losses, episode_returns, episode_trade_gains)
        
        # Append metrics for each episode
        episode_rewards.append(total_reward)
        num_trades.append(trade_count)
        win_rates.append(win_rate)
        sharpe_ratios.append(sharpe_ratio)
        avg_percent_gains.append(avg_percent_gain)
        max_drawdowns.append(max_drawdown)
        total_profits.append(total_profit)
        
        # Replay and update epsilon
        agent.replay()
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward / 1000}, Epsilon: {agent.epsilon:.4f}")
        print(f"Total Profit: {total_profit:.2f}, Total Value: {env.current_total_value:.2f}")
        print(f"Trades: {trade_count}, Win Rate: {win_rate:.2f}, Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    plot_metrics(episode_rewards, num_trades, win_rates, sharpe_ratios, avg_percent_gains, max_drawdowns, total_profits)
    
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
    p.strip_dirs().sort_stats('cumtime').print_stats(20)