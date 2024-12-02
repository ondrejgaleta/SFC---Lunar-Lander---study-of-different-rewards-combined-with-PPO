#!/bin/bash

# Update and upgrade system packages
echo "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install required system packages
echo "Installing dependencies for pygame and box2d..."
sudo apt-get install -y \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libfreetype6-dev libportmidi-dev libjpeg-dev build-essential \
    python3-setuptools python3-dev python3-numpy python3-pip \
    python3-venv cython3 python3-full swig cmake zlib1g-dev

# Set up a virtual environment
echo "Setting up a virtual environment..."
python3 -m venv lunarlander_env
source lunarlander_env/bin/activate

sudo pip install -r requirements.txt --break-system-packages

# Install pygame from GitHub
echo "Installing pygame from source..."
git clone https://github.com/pygame/pygame.git
cd pygame
sudo python3 setup.py build
sudo python3 setup.py install
cd ..

# Install pybox2d from GitHub
echo "Installing pybox2d from GitHub..."
git clone https://github.com/pybox2d/pybox2d.git
cd pybox2d
sudo python3 setup.py build
sudo python3 setup.py install
cd ..


# sudo pip install torch --break-system-packages # chose this option for GPU training
sudo pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# Deactivate the virtual environment
echo "Deactivating the virtual environment..."
deactivate

echo "Setup possibly complete!"
