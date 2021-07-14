# TCIR-Benchmark
This is the official repository for the paper [Benchmarking Tropical Cyclone Rapid Intensification with Satellite Images and Attention-based Deep Models](https://arxiv.org/abs/1909.11616).
You can find the code to reproduce all the experiment results, including 4 ablation study models for rapid intensification prediction.

Model | vanilla | CCA | SSA | CCA + SSA
--- | --- | --- | --- | ---
PR-AUC | 0.951 | 0.963 | 0.951 | 0.961
Heidke skill score | 0.159 | 0.164 | 0.161 | 0.152 

## Requirements
You can install the recommended environment as follows:
```
conda env create -f env.yml -n cyclone
```
The pretrained model weights needed to be combined as follows:
```
cd ./pretrained_models
chmod +x combine.sh
./combine.sh
```

## Quick Start

### Data preparation
The data needs to first be downloaded from [here](https://www.csie.ntu.edu.tw/~htlin/program/TCRISI/).
The data paths in `config.yaml` needs to then be updated according to the path of the data.
The config files are located in the specific model directories in `./pretrained_models`.

### Train
To train the model, run
```
python train.py
```
### Predict
To evaluate model performance, run
```
python predict.py [pretrained_model_dir] [model_1,model_2,...,model_n]
```
For instance to evaluate the pretrained models, run
```
python predict.py --model_dir ./pretrained_models --models ConvLSTM,ConvLSTM_CCA,ConvLSTM_SSA,ConvLSTM_CCA_SSA
```

## Citation
Please cite our work if you use this repo.
```
@inproceedings{bai2020tcri,
  author = {Ching-Yuan Bai and Buo-Fu Chen and Hsuan-Tien Lin},
  title = {Benchmarking Tropical Cyclone Rapid Intensification
                  with Satellite Images and Attention-based Deep
                  Models},
  booktitle = {Proceedings of the European Conference on
                  Machine Learning and Principles and Practice of
                  Knowledge Discovery in Databases (ECML/PKDD)},
  month = sep,
  year = 2020,
  data = {http://www.csie.ntu.edu.tw/~htlin/program/TCRISI},
  pdf = {http://www.csie.ntu.edu.tw/~htlin/paper/doc/ecml20tcrisi.pdf},
  preliminary = {A preliminary version appeared in the Workshop on
                  Machine Learning for Earth Observation @ ECML/PKDD
                  '19.}
}
```
