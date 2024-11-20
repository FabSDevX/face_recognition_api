#!/bin/bash

# Asegúrate de tener las actualizaciones más recientes
apt-get update

# Descarga e instala CMake
apt-get install -y cmake

# Verifica la instalación de CMake
cmake --version
