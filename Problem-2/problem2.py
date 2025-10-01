# import tensorflow as tf
# from tensorflow.keras import layers
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import requests
# import io
# import os

# # --- 1. Data Loading and Preprocessing ---
# def load_and_preprocess_data():
#     """
#     Downloads the eICU age data, scales it to [-1, 1],
#     and returns it with the scaler parameters.
#     """
#     print("Downloading dataset...")
#     # URL for the raw .npy file (THIS LINE HAS BEEN CORRECTED)
#     url = 'https://raw.githubusercontent.com/JeffersonLab/jlab_datascience_data/main/eICU_age.npy'
#     response = requests.get(url)
#     response.raise_for_status()  # Check for download errors
    
#     real_ages = np.load(io.BytesIO(response.content))
#     real_ages = real_ages.astype('float32').reshape(-1, 1)

#     # Scale data to the [-1, 1] range, which is ideal for the tanh activation in the generator
#     min_age = real_ages.min()
#     max_age = real_ages.max()
#     scaled_ages = (real_ages - min_age) / (max_age - min_age) * 2 - 1

#     print(f"Dataset loaded. Original age range: {min_age:.1f} to {max_age:.1f}")
#     return scaled_ages, min_age, max_age

# # --- 2. Model Definition ---
# LATENT_DIM = 100 # The dimension of the random noise input for the generator

# def create_generator():
#     """Defines the Generator model architecture."""
#     model = tf.keras.Sequential([
#         layers.Input(shape=(LATENT_DIM,)),
#         layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2)),
#         layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2)),
#         layers.Dense(1, activation='tanh') # tanh activation scales output to [-1, 1]
#     ], name="generator")
#     return model

# def create_discriminator():
#     """Defines the Discriminator model architecture."""
#     model = tf.keras.Sequential([
#         layers.Input(shape=(1,)),
#         layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2)),
#         layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2)),
#         layers.Dense(1, activation='sigmoid') # Sigmoid for real vs. fake classification
#     ], name="discriminator")
#     return model

# # --- 3. Training Coordination ---
# class GAN(tf.keras.Model):
#     def __init__(self, discriminator, generator, latent_dim):
#         super().__init__()
#         self.discriminator = discriminator
#         self.generator = generator
#         self.latent_dim = latent_dim

#     def compile(self, d_optimizer, g_optimizer, loss_fn):
#         super().compile()
#         self.d_optimizer = d_optimizer
#         self.g_optimizer = g_optimizer
#         self.loss_fn = loss_fn
#         self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
#         self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

#     @property
#     def metrics(self):
#         return [self.d_loss_metric, self.g_loss_metric]

#     @tf.function
#     def train_step(self, real_data):
#         batch_size = tf.shape(real_data)[0]
        
#         # --- Train the Discriminator ---
#         # Goal: Maximize log(D(x)) + log(1 - D(G(z)))
#         random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
#         generated_data = self.generator(random_latent_vectors)
#         combined_data = tf.concat([real_data, generated_data], axis=0)
#         labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        
#         with tf.GradientTape() as tape:
#             predictions = self.discriminator(combined_data)
#             d_loss = self.loss_fn(labels, predictions)
#         grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
#         self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

#         # --- Train the Generator ---
#         # Goal: Maximize log(D(G(z)))
#         random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
#         misleading_labels = tf.ones((batch_size, 1))
#         with tf.GradientTape() as tape:
#             predictions = self.discriminator(self.generator(random_latent_vectors))
#             g_loss = self.loss_fn(misleading_labels, predictions)
#         grads = tape.gradient(g_loss, self.generator.trainable_weights)
#         self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

#         self.d_loss_metric.update_state(d_loss)
#         self.g_loss_metric.update_state(g_loss)
#         return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

# # --- 4. Visualization of Coordination and Final Result ---
# class GANMonitor(tf.keras.callbacks.Callback):
#     """A callback to generate and save plots of the generator's progress."""
#     def __init__(self, real_data_unscaled, latent_dim=LATENT_DIM):
#         self.real_data_unscaled = real_data_unscaled
#         self.latent_dim = latent_dim
#         self.min_val = real_data_unscaled.min()
#         self.max_val = real_data_unscaled.max()
        
#         if not os.path.exists("gan_coordination_progress"):
#             os.makedirs("gan_coordination_progress")

#     def on_epoch_end(self, epoch, logs=None):
#         # Save a progress plot every 20 epochs to illustrate coordination
#         if (epoch + 1) % 20 == 0:
#             random_latent_vectors = tf.random.normal(shape=(1000, self.latent_dim))
#             generated_data_scaled = self.model.generator(random_latent_vectors)
            
#             # Inverse scale the generated data to the original age range
#             generated_ages = (generated_data_scaled + 1) / 2 * (self.max_val - self.min_val) + self.min_val

#             plt.figure(figsize=(10, 6))
#             sns.kdeplot(self.real_data_unscaled.flatten(), label="Real Ages", color='blue', fill=True, lw=0)
#             sns.kdeplot(generated_ages.numpy().flatten(), label="Generated Ages", color='red', fill=True, lw=0)
#             plt.title(f"Coordination Progress: Distribution at Epoch {epoch + 1}")
#             plt.xlabel("Age")
#             plt.ylabel("Density")
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(f"gan_coordination_progress/epoch_{epoch+1:04d}.png")
#             plt.close()

# # --- Main Execution ---
# if __name__ == "__main__":
#     # Hyperparameters
#     EPOCHS = 200
#     BATCH_SIZE = 128
    
#     # 1. Load data
#     scaled_ages, min_age, max_age = load_and_preprocess_data()
#     real_ages_unscaled = (scaled_ages + 1) / 2 * (max_age - min_age) + min_age
#     dataset = tf.data.Dataset.from_tensor_slices(scaled_ages).shuffle(1024).batch(BATCH_SIZE)

#     # 2. Build and compile models
#     discriminator = create_discriminator()
#     generator = create_generator()
#     gan = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
#     gan.compile(
#         d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
#         g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
#         loss_fn=tf.keras.losses.BinaryCrossentropy(),
#     )

#     # 3. Train the model with the monitoring callback
#     print("\nStarting GAN training...")
#     gan.fit(
#         dataset, 
#         epochs=EPOCHS, 
#         callbacks=[GANMonitor(real_ages_unscaled)]
#     )
#     print("Training finished.")
    
#     # 4. Generate the final comparison plot
#     print("Generating final comparison plot...")
#     num_final_samples = real_ages_unscaled.shape[0]
#     random_noise = tf.random.normal(shape=(num_final_samples, LATENT_DIM))
#     generated_data_scaled = generator.predict(random_noise)
#     generated_ages = (generated_data_scaled + 1) / 2 * (max_age - min_age) + min_age
    
#     plt.figure(figsize=(12, 7))
#     sns.histplot(real_ages_unscaled.flatten(), bins=60, kde=True, color='dodgerblue', label='Real Age Distribution', stat='density')
#     sns.histplot(generated_ages.flatten(), bins=60, kde=True, color='darkorange', label='Generated Age Distribution', stat='density')
#     plt.title("Final Comparison: Real vs. Generated Age Distribution", fontsize=16)
#     plt.xlabel("Patient Age", fontsize=12)
#     plt.ylabel("Density", fontsize=12)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.savefig("final_gan_distribution_comparison.png")
#     plt.show()
#     print("Final plot saved as 'final_gan_distribution_comparison.png'.")

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io 

# This import is no longer needed for the modified function
# import requests 

# --- 1. Data Loading and Preprocessing (MODIFIED) ---
def load_and_preprocess_data(file_path='eICU_age.npy'):
    """
    Loads the eICU age data from a local file, scales it to [-1, 1],
    and returns it with the scaler parameters.
    """
    print(f"Loading dataset from local file: {file_path}...")
    
    # This line now loads the file directly from your folder
    real_ages = np.load(file_path)
    
    real_ages = real_ages.astype('float32').reshape(-1, 1)

    # Scale data to the [-1, 1] range, which is ideal for the tanh activation in the generator
    min_age = real_ages.min()
    max_age = real_ages.max()
    scaled_ages = (real_ages - min_age) / (max_age - min_age) * 2 - 1

    print(f"Dataset loaded. Original age range: {min_age:.1f} to {max_age:.1f}")
    return scaled_ages, min_age, max_age

# --- 2. Model Definition ---
LATENT_DIM = 100 # The dimension of the random noise input for the generator

def create_generator():
    """Defines the Generator model architecture."""
    model = tf.keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2)),
        layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2)),
        layers.Dense(1, activation='tanh') # tanh activation scales output to [-1, 1]
    ], name="generator")
    return model

def create_discriminator():
    """Defines the Discriminator model architecture."""
    model = tf.keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2)),
        layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2)),
        layers.Dense(1, activation='sigmoid') # Sigmoid for real vs. fake classification
    ], name="discriminator")
    return model

# --- 3. Training Coordination ---
class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    @tf.function
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        
        # --- Train the Discriminator ---
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_data = self.generator(random_latent_vectors)
        combined_data = tf.concat([real_data, generated_data], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_data)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # --- Train the Generator ---
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

# --- 4. Visualization of Coordination and Final Result ---
class GANMonitor(tf.keras.callbacks.Callback):
    """A callback to generate and save plots of the generator's progress."""
    def __init__(self, real_data_unscaled, latent_dim=LATENT_DIM):
        self.real_data_unscaled = real_data_unscaled
        self.latent_dim = latent_dim
        self.min_val = real_data_unscaled.min()
        self.max_val = real_data_unscaled.max()
        
        if not os.path.exists("gan_coordination_progress"):
            os.makedirs("gan_coordination_progress")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 20 == 0:
            random_latent_vectors = tf.random.normal(shape=(1000, self.latent_dim))
            generated_data_scaled = self.model.generator(random_latent_vectors)
            
            generated_ages = (generated_data_scaled + 1) / 2 * (self.max_val - self.min_val) + self.min_val

            plt.figure(figsize=(10, 6))
            sns.kdeplot(self.real_data_unscaled.flatten(), label="Real Ages", color='blue', fill=True, lw=0)
            sns.kdeplot(generated_ages.numpy().flatten(), label="Generated Ages", color='red', fill=True, lw=0)
            plt.title(f"Coordination Progress: Distribution at Epoch {epoch + 1}")
            plt.xlabel("Age")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"gan_coordination_progress/epoch_{epoch+1:04d}.png")
            plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    # Hyperparameters
    EPOCHS = 5000 #200
    BATCH_SIZE = 200  #128
    
    # 1. Load data
    scaled_ages, min_age, max_age = load_and_preprocess_data()
    real_ages_unscaled = (scaled_ages + 1) / 2 * (max_age - min_age) + min_age
    dataset = tf.data.Dataset.from_tensor_slices(scaled_ages).shuffle(1024).batch(BATCH_SIZE)

    # 2. Build and compile models
    discriminator = create_discriminator()
    generator = create_generator()
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
    gan.compile(
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=tf.keras.losses.BinaryCrossentropy(),
    )

    # 3. Train the model with the monitoring callback
    print("\nStarting GAN training...")
    gan.fit(
        dataset, 
        epochs=EPOCHS, 
        callbacks=[GANMonitor(real_ages_unscaled)]
    )
    print("Training finished.")
    
    # 4. Generate the final comparison plot
    print("Generating final comparison plot...")
    num_final_samples = real_ages_unscaled.shape[0]
    random_noise = tf.random.normal(shape=(num_final_samples, LATENT_DIM))
    generated_data_scaled = generator.predict(random_noise)
    generated_ages = (generated_data_scaled + 1) / 2 * (max_age - min_age) + min_age
    
    plt.figure(figsize=(12, 7))
    sns.histplot(real_ages_unscaled.flatten(), bins=60, kde=True, color='dodgerblue', label='Real Age Distribution', stat='density')
    sns.histplot(generated_ages.flatten(), bins=60, kde=True, color='darkorange', label='Generated Age Distribution', stat='density')
    plt.title("Final Comparison: Real vs. Generated Age Distribution", fontsize=16)
    plt.xlabel("Patient Age", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("final_gan_distribution_comparison.png")
    plt.show()
    print("Final plot saved as 'final_gan_distribution_comparison.png'.")