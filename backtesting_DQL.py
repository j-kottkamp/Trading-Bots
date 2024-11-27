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
import tensorflow as tf

# Häufiger Trainieren e.g. alle 10 Tage
# GPU Training


log_dir = "logs/profile/"
writer = tf.summary.create_file_writer(log_dir)

print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
print("Is cuDNN available: ", tf.test.is_built_with_cuda())

# DQN Agent
class PrioritizedReplayBuffer:
    def __init__(self, size, alpha, priority_threshold=1e-5):
        self.size = size
        self.alpha = alpha
        self.priority_threshold = priority_threshold
        self.buffer = []
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.pos = 0
        self.total_priority_sum = 0  # Running sum of priorities

    def add(self, experience, priority):
        state, action, reward, next_state, done = experience
        # Ensure states and next_states are arrays with proper shape
        state = np.asarray(state)
        next_state = np.asarray(next_state)
        # Check if buffer is full
        if len(self.buffer) < self.size:
            self.buffer.append(experience)
        else:
            # Find the index of the lowest priority to replace it
            min_priority_idx = np.argmin(self.priorities)
            self.total_priority_sum -= self.priorities[min_priority_idx]  # Adjust running sum
            self.buffer[min_priority_idx] = experience
            self.pos = min_priority_idx

        # Add experience and update priority
        adjusted_priority = max(priority, self.priority_threshold)
        self.priorities[self.pos] = adjusted_priority ** self.alpha
        self.total_priority_sum += self.priorities[self.pos]
        self.pos = (self.pos + 1) % self.size

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], []

        # Normalize priorities to probabilities
        probabilities = self.priorities[:len(self.buffer)] / self.total_priority_sum
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices

    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            if priority >= self.priority_threshold:
                # Only adjust the priorities of sampled experiences
                old_priority = self.priorities[idx]
                adjusted_priority = max(priority, self.priority_threshold) ** self.alpha
                self.priorities[idx] = adjusted_priority

                # Update the running total priority sum
                self.total_priority_sum += adjusted_priority - old_priority




class DQNAgent:
    def __init__(self, state_size, action_size, env):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(size=10000, alpha=0.75)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = (self.epsilon_min / self.epsilon) ** (1 / TRAINING_LENGTH) - 0.001
        self.learning_rate = 0.005
        self.batch_size = 1024
        self.env = env  # Store the environment instance
        self.current_position = self.env.current_position  # Initialize with the current position from the environment
        self.model = self.build_model()
    
    def remember(self, state, action, reward, next_state, done):
        priority = abs(reward)
        self.memory.add((state, action, reward, next_state, done), priority)
    
    def build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=(self.env.time_steps, self.state_size)),  # Time steps and features
            layers.GRU(64, return_sequences=True, activation='relu'),
            layers.GRU(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)  # Shape: (1, time_steps, features)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        minibatch, indices = self.memory.sample(self.batch_size)
        states = np.array([exp[0] for exp in minibatch])  # Shape: (batch_size, time_steps, features)
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])  # Shape: (batch_size, time_steps, features)
        dones = np.array([exp[4] for exp in minibatch])

        # Predict Q-values for current and next states
        target_q_values = rewards + self.gamma * (1 - dones) * np.amax(
            self.model.predict(next_states, verbose=0), axis=1
        )
        current_q_values = self.model.predict(states, verbose=0)
        
        for i, action in enumerate(actions):
            current_q_values[i][action] = target_q_values[i]

        self.model.fit(states, current_q_values, epochs=1, verbose=0)

        errors = np.abs(target_q_values - current_q_values[np.arange(len(actions)), actions])
        self.memory.update_priorities(indices, errors)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)





class TradingEnvironment:
    def __init__(self, price_data, initial_investment=10000, time_steps=30):
        self.price_data = price_data
        self.time_steps = time_steps
        self.price_data = price_data
        self.current_step = 0
        self.initial_investment = initial_investment
        self.current_balance = initial_investment
        self.previous_total_value = self.initial_investment
        self.current_total_value = self.initial_investment
        self.shares_held = 0
        self.short_positions = 0
        self.current_position = None
        self.investment_percentage = 0.1
        self.preprocessed_data = self.compute_indicators()
        self.total_profit = 0
        self.total_reward = 0
        self.peak_trade_value = self.initial_investment
        self.max_drawdown_in_trade = 0
        


    def compute_indicators(self):
        close_prices = self.price_data['Close'].values.flatten()
        high_prices = self.price_data['High'].values.flatten()
        low_prices = self.price_data['Low'].values.flatten()
        volume = self.price_data['Volume'].values.flatten()

        indicators = np.array([
            close_prices,
            high_prices,
            low_prices,
            volume,
            
            # Trend Indicators
            ta.EMA(close_prices, timeperiod=9),
            ta.EMA(close_prices, timeperiod=50),
            ta.DEMA(close_prices, timeperiod=30),
            ta.SMA(close_prices, timeperiod=20),
            ta.MOM(close_prices, timeperiod=10),
            ta.T3(close_prices, timeperiod=5),
            ta.TEMA(close_prices, timeperiod=20),
            
            # Momentum Indicators
            ta.RSI(close_prices, timeperiod=14),
            ta.STOCH(high_prices, low_prices, close_prices, fastk_period=14, slowk_period=3, slowd_period=3)[0], # Stochastic %K
            ta.STOCHRSI(close_prices, timeperiod=14, fastk_period=5, fastd_period=3)[0], # Stochastic RSI %K
            ta.WILLR(high_prices, low_prices, close_prices, timeperiod=14),
            ta.CCI(high_prices, low_prices, close_prices, timeperiod=14),
            ta.CMO(close_prices, timeperiod=14),
            ta.ROC(close_prices, timeperiod=10),
            ta.TRIX(close_prices, timeperiod=15),
            
            # Volatility Indicators
            ta.ATR(high_prices, low_prices, close_prices, timeperiod=14),
            ta.NATR(high_prices, low_prices, close_prices, timeperiod=14),
            ta.STDDEV(close_prices, timeperiod=14),
            ta.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)[0], # Upper Bollinger Band
            ta.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)[1], # Middle Bollinger Band
            ta.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)[2], # Lower Bollinger Band
                        
            # Cycle Indicators
            ta.HT_DCPHASE(close_prices),
            ta.HT_DCPERIOD(close_prices),
            ta.HT_PHASOR(close_prices)[0], # HT Phasor Component InPhase
            ta.HT_SINE(close_prices)[0], # Sine
            ta.HT_TRENDLINE(close_prices),
            ta.HT_TRENDMODE(close_prices),
            
            # Price Transformation
            ta.LINEARREG(close_prices, timeperiod=14), # Linear Regression
            ta.LINEARREG_ANGLE(close_prices, timeperiod=14),
            ta.LINEARREG_INTERCEPT(close_prices, timeperiod=14),
            ta.LINEARREG_SLOPE(close_prices, timeperiod=14),
            
            # Statistical Functions
            ta.VAR(close_prices, timeperiod=5),
            ta.TSF(close_prices, timeperiod=14), # Time Series Forecast
            ta.CORREL(high_prices, low_prices, timeperiod=10), # Pearson Correlation Coefficient
            ta.BETA(high_prices, low_prices, timeperiod=5) # Beta
        ]).T
        return indicators

    def reset(self):
        self.current_step = 0
        self.current_balance = self.initial_investment
        self.shares_held = 0
        self.short_positions = 0
        self.current_position = None
        self.total_profit = 0
        self.total_reward = 0
        self.previous_total_value = self.initial_investment
        self.current_total_value = self.initial_investment
        return self.get_state()

    def get_state(self):
        if self.current_step < self.time_steps:
            return np.zeros((self.time_steps, self.preprocessed_data.shape[1]))  # Shape: (time_steps, features)
        return self.preprocessed_data[self.current_step - self.time_steps:self.current_step]  # Ensure this returns (time_steps, features)

    def step(self, action):
        # Current price of the asset
        current_price = self.preprocessed_data[self.current_step][0]
        

        # Calculate the amount to invest based on current balance and investment percentage
        invest_amount = self.current_balance * self.investment_percentage
        reward = 0
        pure_profit = 0
        
        def current_total_value():
            self.current_total_value = self.current_balance + (self.shares_held * 100) - (self.short_positions * 100)
            return self.current_total_value
        
        self.previous_total_value = self.current_total_value
        self.current_total_value = current_total_value()
        
        if self.current_total_value > self.peak_trade_value:
            self.peak_trade_value = self.current_total_value

        # Calculate drawdown during the trade, only if a position exists
        if self.current_position is not None:
            drawdown = calculate_drawdown(self.current_total_value, self.peak_trade_value, self.current_position['type'])
            self.max_drawdown_in_trade = max(self.max_drawdown_in_trade, drawdown)
        else:
            drawdown = 0

        # Actions: Hold, Buy, Sell, Short, Cover
        if action == 0:  # Hold
            reward = self.calculate_reward(0) 

        elif action == 1:  # Buy
            num_shares_to_buy = invest_amount // current_price
            if num_shares_to_buy > 0:
                reward = self.calculate_reward(0.3)
                self.shares_held += num_shares_to_buy
                self.current_balance -= num_shares_to_buy * current_price
                self.current_position = {
                    'type': 'buy',
                    'price': current_price,
                    'amount': num_shares_to_buy
                }
            

        elif action == 2:  # Sell
            if self.shares_held > 0 and self.current_position and self.current_position['type'] == 'buy':
                pure_profit = (current_price - self.current_position['price']) * self.current_position['amount']
                self.total_profit += pure_profit
                self.current_balance += self.shares_held * current_price
                self.shares_held = 0
                reward = self.calculate_reward(pure_profit) 
                self.total_reward += reward
                self.current_position = None
                # Reset peak_trade_value and max_drawdown_in_trade after closing the trade
                self.peak_trade_value = self.current_total_value
                self.max_drawdown_in_trade = 0
            else:
                reward = -0.2 # punish Sell if not 
                

        elif action == 3:  # Short
            num_shares_to_short = invest_amount // current_price
            if num_shares_to_short > 0:
                reward = self.calculate_reward(0.3)
                self.short_positions += num_shares_to_short
                self.current_balance += num_shares_to_short * current_price
                self.current_position = {
                    'type': 'short',
                    'price': current_price,
                    'amount': num_shares_to_short
                }
                

        elif action == 4:  # Cover
            if self.short_positions > 0 and self.current_position and self.current_position['type'] == 'short':
                pure_profit = (self.current_position['price'] - current_price) * self.current_position['amount']
                self.total_profit += pure_profit
                self.current_balance -= self.short_positions * current_price
                self.short_positions = 0
                reward = self.calculate_reward(pure_profit)
                self.current_position = None
                # Reset peak_trade_value and max_drawdown_in_trade after closing the trade
                self.peak_trade_value = self.current_total_value
                self.max_drawdown_in_trade = 0
            else:
                reward = -0.2 # Punish Cover if not shorting

        
        
        self.total_reward += reward
        self.current_step += 1
        done = self.current_step >= len(self.preprocessed_data) - 1
        return self.get_state(), reward, done, self.total_profit, pure_profit

        
    def calculate_reward(self, pure_profit):
        if pure_profit > 0:
            reward = pure_profit
        else:
            reward = -abs(pure_profit)
        return reward


def initialize_environment(price_data, initial_investment=10000):
    """Initialize the trading environment."""
    env = TradingEnvironment(price_data, initial_investment)
    return env

def initialize_agent(env):
    """Initialize the DQN agent based on the environment's state size."""
    state_size = env.preprocessed_data.shape[1]  # This should be 39 if indicators are correct
    action_size = 5  # Buy, Sell, Hold, Short, Cover
    agent = DQNAgent(state_size, action_size, env)
    return agent

def calculate_drawdown(current_value, peak_value, position_type):
    if position_type == 'short':
        drawdown = (current_value - peak_value) / peak_value if peak_value > 0 else 0
    else:  # For long positions
        drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
    return drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """Calculate the Sharpe ratio based on portfolio returns."""
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return (mean_return - risk_free_rate) / std_return if std_return != 0 else 0

def calculate_trade_metrics(reward, current_value, trade_action, episode_trade_gains):
    """Track trade-related metrics."""
    if trade_action in [2, 4]:  # Only count for actual trades
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
    replay_number = 0
    
    while not done:
        current_state = env.get_state()
        action = agent.choose_action(current_state)
        next_state, reward, done, cum_profit, single_profit = env.step(action)
        agent.remember(current_state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        replay_number += 1
        if replay_number % 5 == 0:
            agent.replay()


        # Track wins/losses and returns for Sharpe ratio
        if single_profit > 0: wins += 1
        elif reward < 0: losses += 1
        trade_count = wins + losses  
        episode_returns.append(reward)

        # Track trade gain and drawdown
        episode_trade_gains = calculate_trade_metrics(single_profit, env.current_total_value, action, episode_trade_gains)
        current_value = env.current_total_value
        peak_value = max(peak_value, current_value)
        max_drawdown = max(max_drawdown, calculate_drawdown(current_value, peak_value, position_type="long"))

    return total_reward, trade_count, wins, losses, episode_returns, episode_trade_gains, max_drawdown, cum_profit

def log_metrics(env, initial_investment, trade_count, wins, losses, episode_returns, episode_trade_gains):
    """Log and calculate metrics such as win rate, Sharpe ratio, and profit."""
    # Win rate
    win_rate = wins / trade_count if trade_count > 0 else 0

    # Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(episode_returns)

    # Average percentage gain per trade
    avg_percent_gain = np.mean(episode_trade_gains) if episode_trade_gains else 0

    return win_rate, sharpe_ratio, avg_percent_gain

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

def dqn_training(price_data, episodes, initial_investment=10000):
    
    """Main function for DQN training over multiple episodes."""
    env = initialize_environment(price_data, initial_investment)
    agent = initialize_agent(env)

    # Metrics tracking
    episode_rewards, num_trades, win_rates, sharpe_ratios, avg_percent_gains, max_drawdowns, total_profits = [], [], [], [], [], [], []
    start_time = time.time()
    
    for e in range(episodes):
        ss = f"----------------------- Episode {e+1}/{episodes} -----------------------\n"
        total_reward, trade_count, wins, losses, episode_returns, episode_trade_gains, max_drawdown, total_profit = process_episode(env, agent, initial_investment)
        
        win_rate, sharpe_ratio, avg_percent_gain = log_metrics(env, initial_investment, trade_count, wins, losses, episode_returns, episode_trade_gains)
        
        # Append metrics for each episode
        episode_rewards.append(total_reward)
        num_trades.append(trade_count)
        win_rates.append(win_rate)
        sharpe_ratios.append(sharpe_ratio)
        avg_percent_gains.append(avg_percent_gain)
        max_drawdowns.append(max_drawdown)
        total_profits.append(total_profit)
        
        # Replay and update epsilon
        # agent.replay()
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        print(f"{ss}Total Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.4f}")
        print(f"Total Profit: {total_profit/10:.1f}%")
        print(f"Trades: {trade_count}, Win Rate: {win_rate:.2f}, Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}\n")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    
    plot_metrics(episode_rewards, num_trades, win_rates, sharpe_ratios, avg_percent_gains, max_drawdowns, total_profits)
    
    agent.save("D:/Python Codes/Python/DQN_Stuff/dqn_trading.weights.h5")



# Download price data and run the training
TRAINING_LENGTH = 10
def main():
    SYMBOL = "SPY"
    START = datetime.datetime(2021, 1, 1)
    END = datetime.datetime(2022, 1, 1)
    Training_Length = TRAINING_LENGTH
    price_data = yf.download(SYMBOL, start=START, end=END)
    dqn_training(price_data, Training_Length)

# Profile the main function
cProfile.run('main()', 'profile_output')

# View the results in a human-readable form
with open('D:/Python Codes/Python/DQN_Stuff/bottleNeckDQL.txt', 'w') as f:
    p = pstats.Stats('profile_output', stream=f)
    p.strip_dirs().sort_stats('cumtime').print_stats(20)
