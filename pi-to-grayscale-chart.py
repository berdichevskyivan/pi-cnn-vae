import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2
from scipy.ndimage import sobel
from scipy.signal import correlate2d

# File path
file_path = 'pi_digits.txt'

# Read the digits from the file
with open(file_path, 'r') as file:
    pi_digits = file.read().strip()

# Convert the digits to a list of integers
pi_digits = [int(digit) for digit in pi_digits if digit.isdigit()]

# Determine the size of the matrix (let's aim for a square matrix)
num_digits = len(pi_digits)
matrix_size = int(np.floor(np.sqrt(num_digits)))

# Ensure we have a number of digits that can form a perfect square matrix
pi_digits = pi_digits[:matrix_size**2]

# Create the matrix
matrix = np.array(pi_digits).reshape(matrix_size, matrix_size)

# Scale the matrix to 0-255 for grayscale image
matrix = (matrix / 9) * 255

# Plot the grayscale image
plt.imshow(matrix, cmap='gray')
plt.title(f"Grayscale Image from Pi Digits ({matrix_size}x{matrix_size})")
plt.show()

# Histogram Analysis
plt.hist(matrix.ravel(), bins=256, range=(0, 255), fc='k', ec='k')
plt.title("Histogram of Pixel Intensities")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# Fourier Transform
f_transform = fft2(matrix)
f_transform_shifted = np.fft.fftshift(f_transform)

# Magnitude Spectrum
magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum (Fourier Transform)")
plt.show()

# Apply Sobel filter to detect edges
edges = sobel(matrix)

# Plot the edges
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection (Sobel Filter)")
plt.colorbar()
plt.show()

# Compute autocorrelation
autocorr = correlate2d(matrix, matrix, mode='same')

# Plot the autocorrelation
plt.imshow(autocorr, cmap='gray')
plt.title("Autocorrelation")
plt.colorbar()
plt.show()
