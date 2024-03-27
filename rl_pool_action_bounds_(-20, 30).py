from multiprocessing import Pool

import numpy as np
import torch
import random
import os

from datetime import datetime

from src.system import HydraulicSystem
from src.simulator import Simulator
from src.policy import PolicyREINFORCE, GaussianPDFModel, Optimizer
from src.scenario import MonteCarloSimulationScenario

from IPython.display import clear_output

data_path = './data'
print(os.path.isdir(data_path))

# SET HYPERPARAMS
# GaussianPDFModel Params
n_hidden_layers_policy = 1
dim_hidden_policy = 2
scale_factor_policy = 10.0
std_policy = 0.01
action_bounds_policy = np.array([[-20., 30]]) # These actions are expanded in comparison with real action

# Optimizer params
opt_method_policy = torch.optim.Adam
opt_options_policy = dict(lr=1.0e-1) # 1.0e-2

# MonteCarloSimulationScenario params
N_episodes = 5 # Increasing the number of episodes stabilizes learning, was 5
N_iterations = 300 # was 300
discount_factor = 1.0

# PRINT HYPERPARAMS
print(f'Policy (GaussianPDFModel) Perceptron')
print(f'action_bounds_policy [µm] = {action_bounds_policy}')
print(f'n_hidden_layers_policy = {n_hidden_layers_policy}')
print(f'dim_hidden_policy = {dim_hidden_policy}')
print(f'scale_factor_policy = {scale_factor_policy}')
print(f'std_policy = {std_policy}')
# policy_params = sum(p.numel() for p in model.parameters())
# print(f"Number of policy parameters: {policy_params}")
print()
print('Policy Optimizer')
print(f'opt_method_policy = {opt_method_policy}')
print(f'opt_options_policy = {opt_options_policy}')
print()
print('MonteCarloSimulationScenario')
print(f'N_episodes = {N_episodes}')
print(f'N_iterations = {N_iterations}')
print(f'discount_factor = {discount_factor}')


now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H%M%S")
print(f'Time start: {dt_string}')

def launch(seed):

    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    system = HydraulicSystem(
        jet_length_std=5e-2,
        jet_velocity_std=1e-2,
    )

    simulator = Simulator(
        system, 
        N_steps=10, 
        state_init=np.array([1e3, 0.0]),
    )

    model = GaussianPDFModel(
        dim_observation=system.dim_observation,
        dim_action=system.dim_action,
        dim_hidden=dim_hidden_policy,
        n_hidden_layers = n_hidden_layers_policy,
        scale_factor=scale_factor_policy,
        std=std_policy,
        action_bounds=action_bounds_policy,
    )

    optimizer = Optimizer(
        model=model,
        opt_method=opt_method_policy,
        opt_options=opt_options_policy,
    )

    policy = PolicyREINFORCE(model, optimizer, is_with_baseline=True)

    scenario = MonteCarloSimulationScenario(
        simulator=simulator,
        policy=policy,
        N_episodes=N_episodes,
        N_iterations=N_iterations,
        discount_factor=discount_factor,
        root_data_path=data_path,
        seed=SEED,
        dt_string=dt_string,
    )
    
    try:
        scenario.run()
        clear_output(wait=True)
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
        print('Stop iteration and plot obtained results')
    except AssertionError as e:
        print('Get error:', e)
        print('Stop solve_ivp and plot obtained results')
    finally:
        previous_learning_curve = scenario.learning_curve
        print(f'Number of already conducted iterations {len(previous_learning_curve)}')
        scenario.plot_data()


if __name__ == '__main__':
    
    seed_list = list(range(1,15))
    print(seed_list)

    with Pool(14) as p:
        print(p.map(launch, seed_list))