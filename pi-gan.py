import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

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

# Reshape the data for the GAN (add channel dimension)
matrix = matrix.reshape(-1, matrix_size, matrix_size, 1)

# Parameters
latent_dim = 100
epochs = 10000
batch_size = 32

# Build the generator
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(128 * (matrix_size // 4) * (matrix_size // 4), activation="relu", input_dim=latent_dim))
    model.add(layers.Reshape(((matrix_size // 4), (matrix_size // 4), 128)))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(1, kernel_size=3, padding="same"))
    model.add(layers.Activation("sigmoid"))
    return model

# Build the discriminator
def build_discriminator(matrix_size):
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=3, strides=2, input_shape=(matrix_size, matrix_size, 1), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Compile the discriminator
discriminator = build_discriminator(matrix_size)
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# Build the generator
generator = build_generator(latent_dim)

# The GAN model
z = layers.Input(shape=(latent_dim,))
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)
combined = models.Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# Training the GAN
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    # Train discriminator
    idx = np.random.randint(0, matrix.shape[0], batch_size)
    real_imgs = matrix[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_imgs = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(real_imgs, real)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = combined.train_on_batch(noise, real)

    if epoch % 1000 == 0 or epoch == epochs - 1:
        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]*100}] [G loss: {g_loss}]")

        # Save generated images
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
