import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.losses import binary_crossentropy

# Custom layer to wrap the flatten function
class FlattenLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.keras.layers.Flatten()(inputs)

# Custom VAE loss layer
class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, matrix_size, **kwargs):
        self.matrix_size = matrix_size
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        encoder_inputs, vae_outputs, z_mean, z_log_var = inputs
        flatten_layer = FlattenLayer()
        reconstruction_loss = tf.reduce_mean(binary_crossentropy(flatten_layer(encoder_inputs), flatten_layer(vae_outputs)))
        reconstruction_loss *= self.matrix_size * self.matrix_size
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        return tf.reduce_mean(reconstruction_loss + kl_loss)

# Read the digits from the file (replace with your path)
file_path = 'pi_digits.txt'

with open(file_path, 'r') as file:
    pi_digits = file.read().strip()

pi_digits = [int(digit) for digit in pi_digits if digit.isdigit()]

# Determine the size of the matrix (let's aim for a square matrix)
num_digits = len(pi_digits)
matrix_size = int(np.floor(np.sqrt(num_digits)))

pi_digits = pi_digits[:matrix_size**2]

# Create the matrix
matrix = np.array(pi_digits).reshape(matrix_size, matrix_size)

# Normalize the matrix to the range 0-1
matrix = matrix / 9.0

# Reshape the data for the VAE (add channel dimension)
matrix = matrix.reshape(matrix_size, matrix_size, 1)
matrix = np.expand_dims(matrix, axis=0)  # Add batch dimension

# VAE model parameters
latent_dim = 2

# Encoder
encoder_inputs = tf.keras.Input(shape=(matrix_size, matrix_size, 1))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_inputs)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation='relu')(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(16, activation='relu')(decoder_inputs)
x = layers.Dense((matrix_size // 4) * (matrix_size // 4) * 64, activation='relu')(x)
x = layers.Reshape((matrix_size // 4, matrix_size // 4, 64))(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoder_outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Models
encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = models.Model(decoder_inputs, decoder_outputs, name='decoder')
vae_outputs = decoder(encoder(encoder_inputs)[2])

# Define VAE model
vae = models.Model(encoder_inputs, vae_outputs, name='vae')

# Define the VAE loss layer
vae_loss_layer = VAELossLayer(matrix_size=matrix_size)

# Optimizer
optimizer = tf.keras.optimizers.Adam()

# Custom training loop
epochs = 50
batch_size = 1

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        encoder_inputs_tensor = tf.convert_to_tensor(matrix, dtype=tf.float32)
        z_mean, z_log_var, z = encoder(encoder_inputs_tensor)
        vae_outputs = decoder(z)
        loss = vae_loss_layer([encoder_inputs_tensor, vae_outputs, z_mean, z_log_var])
    grads = tape.gradient(loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

# Get the reconstructed output
reconstructed = vae.predict(matrix)

# Plot the original and reconstructed images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(matrix.reshape(matrix_size, matrix_size), cmap='gray')
plt.title("Original Image")

# Reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(reconstructed.reshape(matrix_size, matrix_size), cmap='gray')
plt.title("Reconstructed Image")

plt.show()

# Generate new samples
n = 10  # number of samples
figure = np.zeros((matrix_size, matrix_size * n))
grid_x = np.linspace(-2, 2, n)
grid_y = np.linspace(-2, 2, n)

for i, yi in enumerate(grid_x):
    z_sample = np.array([[yi, yi]])
    x_decoded = decoder.predict(z_sample)
    digit = x_decoded[0].reshape(matrix_size, matrix_size)
    figure[:, i * matrix_size: (i + 1) * matrix_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.title("Generated Samples")
plt.show()
