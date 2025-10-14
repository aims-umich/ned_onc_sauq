# Sensitivity analysis and uncertainty propagation of the time to onset of natural circulation in air ingress accidents
A repository for the journal paper published in Nuclear Engineering and Design with the title: Sensitivity analysis and uncertainty propagation of the time to onset of natural circulation in air ingress accidents

## Paper

**Extended Version in the PHYSOR 2024 Special Issue (Paper is published as Open-Access):**

Radaideh, M. I., Tunkle, L., Price, D., Abdulraheem, K., Lin, L., & Elias, M. (2025). Multistep Criticality Search and Power Shaping in Nuclear Microreactors with Deep Reinforcement Learning. Nuclear Science and Engineering, 1-13. https://doi.org/10.1080/00295639.2024.2447012

**PHYSOR 2024 Original Conference Paper:** 

Radaideh, M. I., Price, D., Abdulraheem, K., & Elias, M. (2024, April). Demonstration of Microreactor Reactivity Control with Reinforcement Learning. In International Conference on Physics of Reactors (PHYSOR 2024), American Nuclear Society (pp. 1427-1436). https://www.ans.org/pubs/proceedings/article-55612/

## 🛠️ Environment Installation

To set up the environment for this project, follow these steps:

```bash
# 1. Create a new conda environment with name (`rlmicro`) using an external YAML file that contains all packages with the right versions
conda env create -f rlmicro.yml

# 2. Activate the environment
conda activate rlmicro

# 3. Install Jupyter and Papermill for notebook execution if needed
pip install jupyter papermill

```

## How to generate the results

- `NPIC_EMD`: Contains the neural network surrogate models based on Serpent burnup calculations, which are described in the two papers. These surrogate models will be loaded from pre-trained images built using keras/TF and are used to make predictions of `keff` and `QPTR` based on the drum angles (theta1, ..., theta6).  **Leave this directory untouched**.

- `source`: Contains some source Python scripts used by the RL training scripts, which basically includes the gym environment built for the microreactor model under `./source/env.py`.  **You may leave this directory untouched as well**. 

- Step 1a (PPO policy training): Make sure you have 20 cores available or change the `num_cpu` variable in `ppo_control.py` to fit the number of cores you have. This script runs on CPU as we will assume you have no GPU available. You may configure the script to run on GPU on your own by changing the argument `device` in the PPO class. **Then run the script below which will take about 60 min to complete training**. 

  ```bash
  python ppo_control.py
  ```

  Results will be saved to the directory `./ppo_nsteps_300` which contains 4 main files as follows:

  - `ppo_best_model.pkl`: The best PPO model image found during training based on epoch mean reward. 
  - `ppo_last_model.pkl`: The last PPO model image found by the last epoch of the training (not necessarily the best). 
  - `ppo_log.csv`: A CSV logger generated during training that logs the important data collected by the agent. 
  - `ppo_plt.png`: A plot of `keff`, `QPTR`, and `Reward` as a function of epochs, which gets updated during training. This plot is generated online using the data from `ppo_log.csv`.

- Step 1b (PPO policy testing): After Step 1a is completed, open `ppo_control.py` and change the variable `run_mode = train` to `run_mode = test`. By default, you will notice that the variable `model_path= './ppo_nsteps_300/ppo_best_model.pkl'` will read the best PPO model image from the result directory created by Step 1a. You may change the path to the last model if you like. Then run the script again which will take a couple of seconds where the policy will be tested: 
  
  ```bash
  python ppo_control.py
  ```
  Results will be saved to a CSV file called `ppo_test.csv` which will be in the same level as `ppo_control.py`.

- Step 2 (A2C Policy Training and Testing): A2C policy follows the exact same instructions as PPO above for training and testing, except that you should use `a2c_control.py` and all results will have `a2c_*` as a prefix instead of `ppo_*`. Note that the paper reports that PPO was significantly better in performance than A2C, so you may skip running this model if you wish. 

## Notes 
  - The directory `paper_results` includes the pre-trained PPO and A2C image files, CSV loggers, and the PNG plots that were reported in the NSE journal paper including the main Figure 4 in the NSE journal paper. Note that we provide these files as a reference to the user but you can regenerate those on your own as described above. We also report some other results from hyperparameter tuning attempts under `./paper_results/other_results_log` that were not promising, also for reference. 
  - *Important Note*: Please note that, due to the nature of reinforcement learning (RL) training—including factors like starting from a suboptimal initial policy, random seeds, and the overall stochasticity of the process—reproducing the exact same policy results reported in the paper can be quite difficult. The authors automated the training process by running 5–10 policies with identical settings and reporting the best-performing one. On average, however, you should still achieve results that are comparable to those in the paper. In particular, you should consistently observe that PPO outperforms A2C, even if your exact numbers differ from ours.    


