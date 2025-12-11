# LayerDAG

[[Paper]](https://arxiv.org/abs/2411.02322)

## Table of Contents

- [Installation](#installation)
- [Train](#train)
- [Sample](#sample)
- [Eval](#eval)
- [Frequently Asked Questions](#frequently-asked-questions)
  * [Q1: libcusparse.so](#q1-libcusparseso)
- [Citation](#citation)

## Installation

```bash
conda create -n LayerDAG python=3.10 -y
conda activate LayerDAG
pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
conda install -c conda-forge cudatoolkit=11.6
conda clean --all -y
pip install dgl==1.1.0+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
pip install tqdm einops wandb pydantic pandas
pip install numpy==1.26.3
```

## Train

To train a LayerDAG model,

```bash
python train.py --config_file configs/LayerDAG/tpu_tile.yaml
CUDA_LAUNCH_BLOCKING=1 python train_circuit.py --config_file configs/LayerDAG/circuit_file.yaml
```
CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=2 python train_circuit.py --config_file configs/LayerDAG/circuit_file.yaml

python train_blif.py --config_file configs/LayerDAG/blif.yaml

The trained model checkpoint will be saved to a file `model_tpu_tile_{time_stamp}.pth`.

## Sample

For sampling and evaluation,

```bash
python sample.py --model_path X
```
python sample_circuit.py --model_path model_circuit_bench_Oct28-09:02:01.pth --batch_size 16

python sample_circuit.py --model_path model_circuit_bench_Nov04-07:00:09.pth --batch_size 16

python convert_pth_to_bench.py \
         --pth_file circuit_bench_samples/val_generated.pth \
         --output_dir generated_bench_files/


where `X` is the file `model_tpu_tile_{time_stamp}.pth` saved above.

## Frequently Asked Questions

### Q1: libcusparse.so

**An error occurs that the program cannot find `libcusparse.so`**, e.g., OSError: libcusparse.so.11: cannot open shared object file: No such file or directory.

To search for the location of it on linux,

```bash
find /path/to/directory -name libcusparse.so.11 -exec realpath {} \;
```

where `/path/to/directory` is the directory you want to search. Assume that the search returns `/home/miniconda3/envs/LayerDAG/lib/libcusparse.so.11`. Then you need to manually specify the environment variable as follows.

```bash
export LD_LIBRARY_PATH=/home/miniconda3/envs/LayerDAG/lib:$LD_LIBRARY_PATH
```

## Citation

```
@inproceedings{li2024layerdag,
    title={Layer{DAG}: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graph Generation},
    author={Mufei Li and Viraj Shitole and Eli Chien and Changhai Man and Zhaodong Wang and Srinivas Sridharan and Ying Zhang and Tushar Krishna and Pan Li},
    booktitle={International Conference on Learning Representations},
    year={2025}
}
```
