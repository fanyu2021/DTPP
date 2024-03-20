# DTPP

This repository contains the code for the ICRA'24 paper:


[**DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving**](https://arxiv.org/abs/2310.05885)

[Zhiyu Huang](https://mczhi.github.io/)<sup>1</sup>, [Peter Karkus](https://karkus.tilda.ws/)<sup>2</sup>, [Boris Ivanovic](https://www.borisivanovic.com/)<sup>2</sup>, [Yuxiao Chen](https://scholar.google.com/citations?user=AOdxmJYAAAAJ&hl=en)<sup>2</sup>, [Marco Pavone](https://scholar.google.com/citations?user=RhOpyXcAAAAJ&hl=en)<sup>2,3</sup>, and [Chen Lv](https://lvchen.wixsite.com/automan)<sup>1</sup>

<sup>1</sup> Nanyang Technological University, <sup>2</sup> NVIDIA Research, <sup>3</sup> Stanford University


## Getting Started
### 1. Configure devkit and envrionment
To begin, please follow these steps:
- Download the [nuPlan dataset](https://www.nuscenes.org/nuplan#download) and set it up as described [here](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html). 
- Install the nuPlan devkit [here](https://nuplan-devkit.readthedocs.io/en/latest/installation.html) (version tested: v1.2.2). 
- Clone this repository and navigate into the folder:
```
git clone https://github.com/MCZhi/DTPP.git && cd DTPP
```
- Activate the environment created when installing the nuPlan-devkit:
```
conda activate nuplan
```
- Install Pytorch:
```
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
- Add the following environment variable to your `~/.bashrc` (you can customize it):
```
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
```

### 2. Data processing
Before training the DTPP model, you need to preprocess the raw data from nuPlan using:
```
python data_process.py \
--data_path nuplan/dataset/nuplan-v1.1/splits/mini \
--map_path nuplan/dataset/maps \
--save_path nuplan/processed_data
```
Three arguments are necessary: ```--data_path``` to specify the path to the stored nuPlan dataset, ```--map_path``` to specify the path to the nuPlan map data, and ```--save_path``` to specify the path to save the processed data.  ```--total_scenarios``` argument to limit the number of scenarios.

### 3. Training
To train the DTPP model, run:
```
python train.py \
--train_set nuplan/processed_data/train \
--valid_set nuplan/processed_data/valid
```
Two arguments are necessary: ```--train_set``` to specify the path to the processed training data and ```--valid_set``` to specify the path to the processed validation data.

Optional training parameters: ```--train_epochs```, ```--batch_size```, and ```--learning_rate```.

### 4. Testing
To test the DTPP planning framework in nuPlan simulation scenarios, use:
```
python run_nuplan_test.py \
--test_type closed_loop_nonreactive_agents \
--data_path nuplan/dataset/nuplan-v1.1/splits/mini \
--map_path nuplan/dataset/maps \
--encoder_path training_log/your/encoder \
--decoder_path training_log/your/decoder 
```
Choose one of the three options ('open_loop_boxes', 'closed_loop_nonreactive_agents', 'closed_loop_reactive_agents') for ```--test_type```, and specify the paths ```--encoder_path``` and  ```--decoder_path```, which point to your trained models (encoder and decoder). Ensure to provide ```--data_path``` and ```--map_path``` arguments as done in the data process step.

Adjust the ```--scenarios_per_type``` to control the number of scenarios tested.

**Make sure the model parameters in ```planner.py``` in ```_initialize_model``` match those used in training.**


## Citation
If you find this project useful in your research, please consider citing:
```BibTeX
@article{huang2023dtpp,
  title={DTPP: Differentiable joint conditional prediction and cost evaluation for tree policy planning in autonomous driving},
  author={Huang, Zhiyu and Karkus, Peter and Ivanovic, Boris and Chen, Yuxiao and Pavone, Marco and Lv, Chen},
  journal={arXiv preprint arXiv:2310.05885},
  year={2023}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact
If you have any questions or suggestions, please feel free to open an issue or contact us (*zhiyu001@e.ntu.edu.sg*).

