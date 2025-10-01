import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Set up hyperparameters
BUFFER_SIZE = 100000
BATCH_SIZE = 100
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Target network update rate
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2  # Delayed policy update frequency

# --- Replay Buffer ---
# Stores transitions and provides random batches for training
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=BUFFER_SIZE):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def store(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=BATCH_SIZE):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            tf.convert_to_tensor(self.state[ind], dtype=tf.float32),
            tf.convert_to_tensor(self.action[ind], dtype=tf.float32),
            tf.convert_to_tensor(self.next_state[ind], dtype=tf.float32),
            tf.convert_to_tensor(self.reward[ind], dtype=tf.float32),
            tf.convert_to_tensor(self.done[ind], dtype=tf.float32)
        )

# --- Actor and Critic Networks ---
# Defines the neural network architectures for TD3
def create_actor(state_dim, action_dim, max_action):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(action_dim, activation="tanh")(x)
    outputs = outputs * max_action
    model = tf.keras.Model(inputs, outputs)
    return model

def create_critic(state_dim, action_dim):
    # State input
    state_input = layers.Input(shape=(state_dim,))
    # Action input
    action_input = layers.Input(shape=(action_dim,))
    # Concatenate state and action
    concat = layers.Concatenate()([state_input, action_input])
    
    x = layers.Dense(256, activation="relu")(concat)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model([state_input, action_input], outputs)
    return model


# --- TD3 Agent Class ---
# Implements the Twin Delayed Deep Deterministic policy gradient algorithm
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Create Actor and Critic networks
        self.actor = create_actor(state_dim, action_dim, max_action)
        self.actor_target = create_actor(state_dim, action_dim, max_action)
        self.actor_target.set_weights(self.actor.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        self.critic_1 = create_critic(state_dim, action_dim)
        self.critic_2 = create_critic(state_dim, action_dim)
        self.critic_target_1 = create_critic(state_dim, action_dim)
        self.critic_target_2 = create_critic(state_dim, action_dim)
        self.critic_target_1.set_weights(self.critic_1.get_weights())
        self.critic_target_2.set_weights(self.critic_2.get_weights())
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=3e-4)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        self.total_it = 0

    def select_action(self, state, exploration_noise=0.1):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = self.actor(state)
        noise = tf.random.normal(action.shape) * exploration_noise
        action = tf.clip_by_value(action + noise, -self.max_action, self.max_action)
        return action.numpy().flatten()

    def train(self):
        self.total_it += 1
        
        # Sample a batch from the replay buffer
        state, action, next_state, reward, done = self.replay_buffer.sample(BATCH_SIZE)
        
        with tf.GradientTape(persistent=True) as tape:
            # Target Policy Smoothing: Add clipped noise to the target actions
            noise = tf.clip_by_value(
                tf.random.normal(action.shape) * POLICY_NOISE, -NOISE_CLIP, NOISE_CLIP
            )
            next_action = tf.clip_by_value(
                self.actor_target(next_state) + noise, -self.max_action, self.max_action
            )

            # Clipped Double-Q Learning: Compute the target Q-value
            target_q1 = self.critic_target_1([next_state, next_action])
            target_q2 = self.critic_target_2([next_state, next_action])
            target_q = tf.minimum(target_q1, target_q2)
            target_q = reward + (1 - done) * GAMMA * target_q

            # Get current Q estimates
            current_q1 = self.critic_1([state, action])
            current_q2 = self.critic_2([state, action])

            # Compute critic loss
            critic_loss_1 = tf.reduce_mean(tf.square(current_q1 - target_q))
            critic_loss_2 = tf.reduce_mean(tf.square(current_q2 - target_q))

        # Optimize the critics
        critic_grads_1 = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)
        self.critic_optimizer_1.apply_gradients(zip(critic_grads_1, self.critic_1.trainable_variables))
        
        critic_grads_2 = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)
        self.critic_optimizer_2.apply_gradients(zip(critic_grads_2, self.critic_2.trainable_variables))
        
        del tape

        # Delayed policy updates
        if self.total_it % POLICY_FREQ == 0:
            with tf.GradientTape() as tape:
                actor_loss = -tf.reduce_mean(self.critic_1([state, self.actor(state)]))
            
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            
            # Soft update target networks
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target_1, self.critic_1)
            self._soft_update(self.critic_target_2, self.critic_2)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.variables, source.variables):
            target_param.assign(target_param * (1.0 - TAU) + param * TAU)

# --- Main Script to Coordinate Agent and Environment ---
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(state_dim, action_dim, max_action)

    episodes = 250
    max_steps_per_episode = 200
    
    episode_rewards = []

    print("Starting Training...")
    for i in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            agent.replay_buffer.store(state, action, next_state, reward, float(done))
            state = next_state
            episode_reward += reward

            # Train agent
            if agent.replay_buffer.size > BATCH_SIZE:
                agent.train()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode: {i+1}, Reward: {episode_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")

    env.close()
    
    # --- Plotting the Learning Behavior ---
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Behavior of TD3 Agent on Pendulum-v1")
    plt.grid(True)
    
    # Calculate and plot a moving average to see the trend
    moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
    plt.plot(np.arange(99, len(episode_rewards)), moving_avg, 'r', linewidth=2, label='100-Episode Moving Average')
    plt.legend()
    
    plt.savefig("td3_pendulum_learning_curve.png")
    print("\nTraining finished. Plot saved as 'td3_pendulum_learning_curve.png'")
    plt.show()