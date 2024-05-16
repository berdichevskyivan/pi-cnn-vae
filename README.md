\# Pi CNN VAE

This repository contains Python scripts for exploring patterns in the seemingly random digits of pi through generative modeling. The scripts implement Generative Adversarial Networks (GANs) and Convolutional Variational Autoencoders (CVAEs) to generate images resembling sequences of pi digits.

## Overview

- `pi-gan.py`: Implements a GAN model to generate images resembling pi digits.
- `pi-cnn-vae.py`: Utilizes a CVAE model to generate images resembling pi digits.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/pi-cnn-vae.git
```

2. Navigate to the cloned directory:

```bash
cd pi-cnn-vae
```

3. Ensure you have installed the required dependencies. You can install them using pip:

```bash
pip install -r requirements.txt
```

4. Execute the desired script:

For GAN:

```bash
python pi-gan.py
```

For CNN-VAE:

```bash
python pi-cnn-vae.py
```

## File Structure

- `pi-gan.py`: GAN script.
- `pi-cnn-vae.py`: CNN-VAE script.
- `pi_digits.txt`: Input file containing pi digits.
- `README.md`: This README file.
- `requirements.txt`: List of required dependencies.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

