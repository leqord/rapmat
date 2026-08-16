# rapmat

Rapid materials discovery TUI tool, based on random crystal generation and machine learning interatomic potentials.

# Installation

An Nvidia GPU is highly recommended.
Linux is recommended as well, since all backends are currently supported on Linux systems.
[Conda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) may be [useful](https://www.anaconda.com/docs/getting-started/working-with-conda/conda-intro-tutorial).

## Linux

Install `pytorch<2.10.0` with CUDA support if you have an NVIDIA GPU, otherwise skip this step:

```bash
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then install rapmat:

```bash
# basic install
pip install rapmat

# mattersim support
pip install rapmat[mattersim]

# all calculators at once
pip install rapmat[all-calculators]
```

Run its TUI:

```bash
rapmat
```

## Windows

On Windows, only the UPET and MatterSim (the latter if built with its prerequisites installed) MLIPs are supported.

One way to overcome the Windows limitations is WSL2 -- check the [Nvidia](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) or [Ubuntu](https://ubuntu.com/wsl/docs/latest/howto/gpu-cuda/) guides. 

### Install WSL2


- Install the [Nvidia Drivers](https://www.nvidia.com/Download/index.aspx)

- Install wsl:
```bash
wsl.exe --install
```

- Ensure it's up to date:
```bash
wsl.exe --update
```

- Enter into WSL (Ubuntu is the default distro):
```bash
wsl.exe
```

- Install CUDA Toolkit *inside* WSL:
```bash
$ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb

$ sudo dpkg -i cuda-keyring_1.1-1_all.deb

$ sudo apt-get update

$ sudo apt-get -y install cuda-toolkit-13-2
```

### Update the linker paths

NequIP compiles its CUDA kernels and needs the CUDA (stub) libraries on the linker path before installation:

```bash
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
```

Or, even better, add it to the `~/.bashrc` to make it permanent:
```bash
echo 'export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Proceed to the linux install guide

# VASP

You need to set the path to VASP pseudopotentials:
```bash
# ~/vasp/pp/potpaw
export VASP_PP_PATH=~/vasp/pp >> ~/.bashrc
source ~/.bashrc
```

# Usage

## Basic concepts

A study defines the system (e.g. Al-O) you are working on and the calculation settings like calculator, forces convergence criterion or pressure.
A run defines a specific `formula x [formula units range]`: e.g. `Al2O3 x 6..8` constituting the unit cell being calculated.

Each run belongs to exactly one study, while a study may have many runs. 
Runs in one study may overlap, but you can view and perform actions such as deduplication or thickness filtering for only one run at a time. 
