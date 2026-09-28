# rapmat

Rapid materials discovery in the terminal.

- [Layout](#layout)
- [Installation](#installation)
- [Test run](#test-run)
- [Concepts](#concepts)
- [Using the TUI](#using-the-tui)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Layout

| Path | Contents |
|---|---|
| `rapmat/core/` | Search, deduplication, phonons, validation, export |
| `rapmat/calculators/` | MLIP and DFT backends |
| `rapmat/storage/` | Database access layer and schema migrations |
| `rapmat/tui/` | Terminal user interface |
| `rapmat/utils/` | Hardware detection, crash-safe spglib calls, other helpers |
| `tests/` | pytest suite |
| `examples/` | Expected output of the [test run](#test-run) |
| `pyproject.toml` | Package metadata, dependencies and optional calculator extras |
| `alembic.ini` | Migration tooling config, developers only |
| `.github/workflows/` | PyPI release workflow |

## Installation

| Requirement | |
|---|---|
| OS | Linux (recommended) runs every calculator. On Windows, use WSL2, or stay on native Windows for MatterSim and UPET only |
| Python | 3.12, in a conda environment |
| GPU | An NVIDIA GPU is highly recommended. Some MLIPs also run on the CPU, much slower |
| Disk space | About 8 GB for the environment with CUDA PyTorch and all calculators, 5 GB more for the CUDA toolkit (NequIP only), plus the models downloaded on first use up to a few GB |

rapmat requires Python 3.12 or later, but it pins PyTorch 2.9, which does not fully support Python 3.14 (it disables `torch.compile` there), so building the NequIP model fails on it. MatterSim is published pre-built only for Linux and macOS on Apple silicon, and only for Python 3.12 and 3.13.

Platform-specific steps:

- **Native Windows**: install the [C++ Build Tools](#windows-native)
- **WSL2**: set up [WSL](#wsl2) first.
- **Linux**: for NequIP, install the [NequIP prerequisites](#nequip-prerequisites-linux-and-wsl).

### 1. Install conda

[Miniforge](https://github.com/conda-forge/miniforge) is recommended, miniconda also works.

Linux and WSL:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Accept the license and the install location, answer `yes` when asked to update your shell profile, then open a new terminal.

Windows: run [`Miniforge3-Windows-x86_64.exe`](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe), or

```powershell
winget install CondaForge.Miniforge3
```

use the **Miniforge Prompt** from the Start menu. To use conda from PowerShell instead, run `conda init powershell` once in the Miniforge Prompt.

### 2. Create the environment

```bash
conda create -n rapmat python=3.12
conda activate rapmat
```

Activate the environment in every new terminal before installing or running rapmat.

### 3. Install PyTorch

Install it before rapmat, otherwise pip takes the default build from PyPI, which on Windows is CPU-only. With an NVIDIA GPU:

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
```

Without an NVIDIA GPU, or with an RTX 50 series card, replace `cu126` in the URL with the index from this table:

| Index | For |
|---|---|
| `cu126` | NVIDIA GPUs from the GTX 900 series (Maxwell) to the RTX 40 series and H100 |
| `cu128` | RTX 50 series (Blackwell) and newer, which `cu126` does not support |
| `cpu` | No NVIDIA GPU |

### 4. Install rapmat

From a clone of this repository or its source archive:

```bash
pip install ".[mattersim,upet]"
```

Or the latest release from PyPI:

```bash
pip install "rapmat[all-calculators]"
```

| Extra | Adds | Platforms |
|---|---|---|
| none | Only VASP | All |
| `mattersim` | MatterSim 5M | Linux, WSL, Windows (compiled during install) |
| `nequip` | NequIP OAM-L | Linux, WSL |
| `upet` | UPET OAM-XL | Linux, WSL, Windows |
| `all-calculators` | All | Linux, WSL |

Combine extras with commas.

### 5. Check the installation

```bash
rapmat
```

The header shows `CUDA` when PyTorch finds a GPU and `CPU` otherwise, and the Home screen shows the GPU name under Hardware. 
Press `i` (Status) and check that the installed calculators are listed as `installed`. 
The TUI also works with the mouse in most terminals.

### Windows (native)

1. Install the [NVIDIA driver](https://www.nvidia.com/Download/index.aspx).
2. MatterSim has no pre-built Windows package, so pip compiles it, which needs the Microsoft C++ Build Tools. Download [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the **Desktop development with C++** workload (MSVC and the Windows SDK), or run:

   ```powershell
   winget install --id Microsoft.VisualStudio.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```

   Open a new terminal afterwards.
3. Follow steps 1 to 5.

### WSL2

NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) and Ubuntu's [WSL GPU guide](https://ubuntu.com/wsl/docs/latest/howto/gpu-cuda/) have more detail.

On Windows:

1. Install the [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) for Windows. WSL uses this driver. **Do not install an NVIDIA driver inside WSL!**
2. Enable WSL and install Ubuntu, restart if asked (run as admin):
   ```powershell
   wsl --install
   ```
   If WSL is already enabled:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
   To update:
   ```powershell
   wsl --update
   ```
3. Start Ubuntu with `wsl` from a normal terminal and create your Linux user. 

Inside WSL (Ubuntu):

1. Check that the GPU is visible:

   ```bash
   nvidia-smi
   ```

2. For NequIP, install the [NequIP prerequisites](#nequip-prerequisites-linux-and-wsl).
3. Follow steps 1 to 5 with the Linux commands.

Under `/mnt/c` file access is slower, keep things in `~` if possible.

### NequIP prerequisites (Linux and WSL)

On first use, it needs a C/C++ compiler and the CUDA toolkit, whose version should match the PyTorch build (12.6 for `cu126`, 12.8 for `cu128`). On Ubuntu in WSL:

```bash
sudo apt update
sudo apt install -y build-essential
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6
```

On native Ubuntu, replace `wsl-ubuntu` with your release, such as `ubuntu2404` or `ubuntu2204`. 
Note, that `cuda-toolkit-12-6` installs only the toolkit, without a driver.

The compile also links against the CUDA driver library, so add the toolkit's stub library to the linker path:

```bash
echo 'export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### VASP (optional)

Point `VASP_PP_PATH` at the directory that contains the POTCAR sets:

```bash
echo 'export VASP_PP_PATH=$HOME/vasp/pp' >> ~/.bashrc
source ~/.bashrc
```

Each POTCAR is found at `$VASP_PP_PATH/<set>/<name>/POTCAR`.

## Test run

The `mattersim` and `upet` extras are necessary. The models are downloaded on first use. This run takes about 1.5 h, depending on the hardware.

1. Home -> `s` (Studies) -> `n` (New): *System* `Si`, a *Study Name* of your choice, *Domain* `bulk`, *Calculator* `MATTERSIM`, *Force conv. crit* `0.01` (the default is `0.005`). Leave *Default symprec* at `0.01`. Move to **Create Study** and press `Enter`, then **OK**.
2. `Enter` on the new study -> `n` (New Run): *Formula* `Si`, *Formula units min* `1` and *max* `8`, *Candidates/group* `2`, *Seed* `1`. `F5` (**Start Run**) queues 8 x 230 x 2 = 3680 candidates. Wait for rapmat to generate and relax them.
3. Answer **Yes** to *View results?*. Structures are sorted by energy per atom. Diamond Si comes first: the leading rows are cells of different size of the same structure, with *Final SG* `Fd-3m (227)` (if *symprec* is kept loose enough) and an energy of about -5.4095 eV/atom.
4. `Esc` twice, back to the study -> `d` (Dedup) on the run -> `F5` (**Analyze**), then **Apply to DB** (`a`) and **OK**. `Esc` twice, back to the study, and `Enter` on the run reopens Results. `d` hides the duplicates.
5. `s` (Save): *Scope* `All N filtered structures`, *Format* `cif`, keep *Results table* at `txt`, then **Save**. The structures and `results_table.txt` are written to `saved_<run>/` in the directory rapmat was started from. The table should roughly match [`examples/expected_results_table.txt`](examples/expected_results_table.txt).
6. `v` (Validate): *Ref. calculator* `UPET`, *Top N* `20` -> `F5` (**Evaluate**). The Validation Results header shows Kendall tau with its p-value and n, and the MAE of MatterSim against UPET over the visible structures. Expect about tau = 0.82 (p = 4.6e-9, n = 20) and an MAE of 0.028 eV/atom. [`examples/expected_eval.txt`](examples/expected_eval.txt) contains the full header and table examples.

The digits can differ with the hardware and library versions.

## Concepts

### Study and run

A **study** fixes the chemical system (e.g. `Al-O`) and the settings shared by all its runs: calculator, domain, external pressure, relaxation convergence criterion, etc.

A **run** belongs to one study and searches a formula over a range of formula units, e.g. `Al2O3 x 2..4`.

## Using the TUI


### Home

Below the menu, Home shows the database location and the detected hardware with the GPU name. The right panel lists the five most recent runs with counts per structure status. Press `Right` to move into it and `Enter` to open a run's results.

### Studies

| Key | Action |
|---|---|
| `n` | New study |
| `Enter` | Open the study |
| `Del` | Delete the study with all its runs and structures, after confirmation |

### Study detail

| Key | Action |
|---|---|
| `Enter` | Results of the focused run |
| `r` | Resume the focused run |
| `n` | New run in this study |
| `d` | Deduplication of the focused run |
| `Del` | Delete the focused run, after confirmation |


### Results

Lists the relaxed structures of one run, sorted by energy per atom (enthalpy under pressure), lowest first.

| Key | Action |
|---|---|
| `Enter` | 3D view |
| `s` | Save structures |
| `o` | Options |
| `v` | Validate against a reference calculator |
| `t` | Maximum thickness |
| `p` | Phonons |
| `d` | Show or hide duplicates |
| `?` | Help |

`Esc` first clears an active search, then goes back.

### Validation

Computes single-point energies of the MLIP-relaxed structures with a reference calculator, typically VASP, and compares the rankings.

The structures passed are those visible in Results when `v` is pressed, including the thickness filter, for example. 

`t` and `d` filters work as in Results.


### DB Settings

The screen shows the active database and the `db.toml` path.

- **Test** opens the database at the path, creating it if needed.
- **Save** switches to the new database.
- **Clear Config** deletes `db.toml` after confirmation. The default location is used from the next start.
- **Disk Usage** shows the database size and how much of it can be reclaimed. 
- **Compact Now** releases that space, after confirmation.

### VASP settings

**TOML file.** The keys are passed to ASE's [`Vasp` calculator](https://ase-lib.org/ase/calculators/vasp.html) as keyword arguments:

```toml
xc = "pbe"
encut = 520
kspacing = 0.25
ismear = 0
sigma = 0.05
```

A `directory` key is ignored, rapmat assigns one per calculation.

## Troubleshooting

**NequIP fails on first use.** The error gives the exit code and the path to `nequip_install.log`. Check the [NequIP prerequisites](#nequip-prerequisites-linux-and-wsl). *CUDA_HOME environment variable is not set* in the log means the CUDA toolkit is missing.

**Space-group labels look wrong, or everything is P1.** Labels depend on symprec. Press `o` in Results or Phase Analysis and relabel with a looser (larger) or tighter (smaller) value.

**Where do saved files go?** Into the directory rapmat was started from, unless you change the path.

## License

See [LICENSE](LICENSE).
