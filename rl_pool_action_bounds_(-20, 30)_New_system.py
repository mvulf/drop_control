from multiprocessing import Pool

import numpy as np
import torch
import random
import os

from datetime import datetime

from src.system import HydraulicSystem
from src.simulator_casadi import create_simulator
from src.policy import PolicyREINFORCE, GaussianPDFModel, Optimizer
from src.scenario import MonteCarloSimulationScenario

import regelum as rg

from IPython.display import clear_output

data_path = './data'
print(os.path.isdir(data_path))

# Initial state
# Define the initial state
p_atm = 1e5
# initial_state = np.array([1e3, 0, 0, p_atm, p_atm])
initial_state = rg.array([1e3, 0, 0, p_atm, p_atm])


# SET HYPERPARAMS
# GaussianPDFModel Params
n_hidden_layers_policy = 1
dim_hidden_policy = 2
scale_factor_policy = 10.0
std_policy = 0.003  # Reduced from 0.005 for more stable learning
action_bounds_policy = np.array([[-20., 30]]) # These actions are expanded in comparison with real action

# Optimizer params
# opt_method_policy = torch.optim.Adam
opt_method_policy = torch.optim.Adam
opt_options_policy = dict(
    lr=6e-3,
    weight_decay=2e-4,  # Added back for better regularization
)

# MonteCarloSimulationScenario params
N_episodes = 10 # Increased from 5 for better stability
N_iterations = 300
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
        init_state=initial_state,
        system_parameters_init = {
            "p_l_gauge": 1.5e5,
            "x_th_limits": (0., 20.),
            "freq_th": 500.0,
            "m_p": 20e-3,
            "D_th": 200e-6, # WAS 200e-6, then 5e-3
            "D_hydr": 20e-3,
            "D_work": 20e-3,
            "h_work_init": 1e3,
            "D_exit": 0.33e-3,
            "l_exit": 8.5e-3,
            "p_coulomb": 10e3, # WAS 10e3, then 1e3
            "eta": 0.70,
            "zeta_th": 5.0, # WAS 5.0, then 0.1
            "rho_hydr": 1e3,
            "rho_work": 1e3,
            "beta_v_hydr": 0.49e-9,
            "beta_v_work": 0.49e-9,
            "sigma_work": 73e-3,
            "mu_work": 1.0e-3,
            "v_j": 200.,
            "jet_length_std": 5e-2,
            "jet_velocity_std": 1e-2,
            "pressure_std": 0.,
            "p_atm": 1e5, # Atmosphere (ambient) pressure, Pa
            "g": 9.81, # gravity constant, m/s^2
        },
    )

    # Create CasADi simulator
    simulator = create_simulator(
        simulator_type='casadi',
        system=system,
        N_steps=10,
        state_init=initial_state,
        integration_method='rk4',  # Change from 'collocation' to 'rk4'
        n_substeps=10000,
        use_symbolic=True
    )

    # MODEL WAS PREVIOUSLY TUNED in `reinforce_3_state_hydraulic_system.ipynb`
    model = torch.load(
        "models/model_policy_reinforce_3_state_system_paper_params.pkl",
        map_location='cpu',
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
    
    seed_list = list(range(1,15)) # ['2', '4', '7', '11', '14']
    print(seed_list)

    with Pool(14) as p: # 5
        print(p.map(launch, seed_list))