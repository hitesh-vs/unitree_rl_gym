#!/usr/bin/env bash
#SBATCH --mail-user=sviswasam@wpi.edu
#SBATCH --mail-type=ALL
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:L40S:1
#SBATCH -t 23:59:00
#SBATCH --mem 128G
#SBATCH --job-name="film"
#SBATCH --output=/home/sviswasam/dr/unitree_rl_gym/logs/output_baseline3.log
#SBATCH --error=/home/sviswasam/dr/unitree_rl_gym/logs/err_baseline3.err

# 1. Load basic modules
source /etc/profile.d/modules.sh
module load cuda12.1/toolkit/12.1.1
module load libx11/1.8.12/wtcqjwl
module load glew/2.2.0/azi6l2x
module load gcc/13.2.0
module load python/3.8.13/slu6jvw

# 2. Set Compiler flags
export CXX=g++
export CC=gcc
export MAX_JOBS=4

# 3. CRITICAL: Explicit Python Headers (the slu6jvw path)
export PYTHON_HEADERS=/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-12.1.0/python-3.8.13-slu6jvwlh43vemachntuxtqyqbxpltdg/include/python3.8
export CPATH=$PYTHON_HEADERS:$CPATH
export C_INCLUDE_PATH=$PYTHON_HEADERS:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$PYTHON_HEADERS:$CPLUS_INCLUDE_PATH

# 4. Add additional include paths for dependencies
export CPATH=$CPATH:/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/mesa-glu-9.0.2-fzzvroqpxdyho2fdiav4wk23ueazizkh/include
export CPATH=$CPATH:/cm/shared/spack/opt/spack/linux-x86_64/libx11-1.8.12-wtcqjwlazka4hmy6zdqc7nmzowe6omgr/include
export CPATH=$CPATH:/cm/shared/spack/opt/spack/linux-x86_64/xproto-7.0.31-lih7ldnzfw22idompow35rqkyqeo6gay/include
export CPATH=$CPATH:/cm/shared/spack/opt/spack/linux-x86_64/glew-2.2.0-azi6l2xafyqw4k4c46rel6bbf5tjon6v/include
export CPATH=$CPATH:/cm/shared/spack/opt/spack/linux-x86_64/mesa-25.0.5-rclmohiaitjskxy2qh4gqxqi2wt5hv75/include

# 5. Library Paths (Prioritize GCC and Local Libs)
GCC_LIB=$(dirname $(gcc --print-file-name=libstdc++.so.6))
export LD_LIBRARY_PATH=$GCC_LIB:/home/sviswasam/dr/local_libs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/llvm-18.1.8-bpqadvig7ku5rfx4jckqwuhf6lk5uljq/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/mesa-glu-9.0.2-fzzvroqpxdyho2fdiav4wk23ueazizkh/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cm/shared/spack/opt/spack/linux-x86_64/mesa-25.0.5-rclmohiaitjskxy2qh4gqxqi2wt5hv75/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cm/shared/spack/opt/spack/linux-x86_64/glew-2.2.0-azi6l2xafyqw4k4c46rel6bbf5tjon6v/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sviswasam/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Suppress common compiler warnings that clutter the log
export CFLAGS="-Wno-implicit-function-declaration -Wno-int-conversion"

# 6. Activate Environment & Prep
source /home/sviswasam/dr/unitree_env/bin/activate

# Force load the pre-compiled gymtorch .so directly
# This bypasses ninja's stale-check entirely
export TORCH_EXTENSIONS_DIR=/home/sviswasam/.cache/torch_extensions/py38_cu121

# 7. Execute
python modular_policy/train_modular.py \
    --xml_path /home/sviswasam/dr/ModuMorph/modular/unitree_g1_actual/xml/g1_12dof_stripped.xml \
    --variants_metadata resources/robots/g1_variants_targeted/variants_metadata.json \
    --num_envs 512 \
    --headless \
    --out_dir ./output_baseline_targeted \
    --seed 1409 \
    --graph_encoding none \
    --resume output_baseline_wide/Mar31_18-20-20/model_400.pt \
    --max_iters 2000