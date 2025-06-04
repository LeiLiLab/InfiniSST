#!/bin/bash

# Script to download and extract model files
echo "Downloading model files..."
wget -O iwslt2025.zip "https://www.dropbox.com/scl/fi/dysdopa9e4l6hghocre1t/iwslt2025.zip?rlkey=2c6uoe3sus55iha90ds5j71r3&dl=1"

echo "Extracting files..."
unzip iwslt2025.zip

echo "Cleaning up..."
rm iwslt2025.zip

echo "Model files downloaded and extracted successfully!" 