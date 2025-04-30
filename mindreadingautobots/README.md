# MindReadingAutobot


### Installation


We use Conda for package management, you will first need to build a conda environment to run the code:


- Build a conda environment with the autobots.yml file (This file is for Linux) 

  ```
  conda env create -f autobots.yaml
  ```
  This will create a conda environment called **autobots** with the packages specified in the file. It will take a while as there are a lot of dependencies. 

- Once the above is done, activate the environment:

  ```
  conda activate autobots
  ```

  Then do a editable install of our own code
  ```
  cd mindautoreadingbots
  pip install -e .
  ```
  
Now the environment is ready, you should be able to generate datasets, launch jobs, and analyze results. 


### Pipeline 
Follow these steps to run experiments and analyze data. 
1. Go to **mindreadingautobots/data/make_datasets.ipynb**, run the corresponding block of code to generate data, this will create a folder of the data in the same directory. For example you generated **sparse_majority_k3_nbits51_n2000_bf24.6_seed1234** 

2. Launch the experiment: 
- Under **mindreadingautobots/hyper_config**, there are some sample configuration YAML files, use one of them or create one for your purpose. Make sure you set **total_gpus** to be 0 as we have not implemented it on GPUs. 
- Activate the conda environment, go to **mindreadingautobots/src/mindreadingautobots** in the terminal, run the command to launch your experiment (see sample.sh)
```python
python -m main -mode tune -dataset sparse_majority_k5_nbits21_n2000_bf35_seed1234 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml \
-model_type RNN -noiseless_validation -epochs 1000\ 
``` 

3. Once the results are finished, it will be available at **mindreadingautobots/src/mindreadingautobots/tune_results**, find the folder with your model and dataset and the time you launched it. You can then analyze the results by using these data. 





























