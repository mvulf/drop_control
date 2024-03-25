import pandas as pd
import numpy as np
import torch

import os

import pickle

from typing import Tuple, Dict, Optional, Callable, Type, Any

from tqdm.notebook import tqdm
from IPython.display import clear_output

from src.simulator import Simulator
from src.policy import PDController

from datetime import datetime

import matplotlib.pyplot as plt

plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "mathptmx",
    'font.size': 14,
})



class SimulationScenario:
    
    def __init__(
        self,
        simulator: Simulator,
        policy,
        root_data_path:str,
        discount_factor: float = 1.0,
        dpi: int = 400,
        seed: int = None,
    ):
        self.simulator = simulator
        self.policy = policy
        
        self.root_data_path = root_data_path
        
        self.discount_factor = discount_factor
        
        self.dpi = dpi
        
        self.seed = seed
        
        self.data_path = None
        
        self.clean_data()
        
        
    def clean_data(self):
        self.observations = []
        self.actions = []
        self.states = []
        self.clean_observations = []
        
        self.total_objective = 0
    
    
    def compute_running_objective(
        self, observation: np.array, action: np.array
    ) -> float:
        """Computes running objective

        Args:
            observation (np.array): current observation
            action (np.array): current action

        Returns:
            float: running objective value
        """
        
        length_diff = (1 - observation[0])

        
        # # WAS
        # # If length smaller than critical, penalty for the negative velocity
        # # If length larger than critical, penalty for the positive velocity
        # return length_diff ** 2 -\
        #     np.sign(length_diff) * observation[1] * abs(observation[1]) * 1 # get relative velocity in [1/ms] # Previous miltiply was 1e2
        
        return length_diff ** 2
    
    
    def compute_total_objective(
        self,
        observations, 
        actions, 
    ):
        total_objective = 0
        
        for step_idx, (observation, action) in enumerate(
            zip(observations, actions)
        ):
            discounted_running_objective = self.discount_factor ** (
                step_idx
            ) * self.compute_running_objective(observation, action)
            # for learning curve plotting
            total_objective += discounted_running_objective
        
        return total_objective
    
    
    def get_real_actions(self, actions:list):

        A = np.zeros((len(actions)*2, 2))

        for i in range(len(actions)-1):
            A[i*2+1:i*2+3, 1] = actions[i+1]
            if i > 0:
                A[i*2:i*2+2, 0] = i

        A[-2:,0] = len(actions) - 1
        
        return A
    
    
    def run(self):
        while self.simulator.step():
            (
                step_idx,
                state,
                observation,
                action,
            ) = self.simulator.get_sim_step_data()

            # PD CONTROLLER
            new_action = self.policy.get_action(observation)  
      
            self.simulator.set_action(new_action)
            self.observations.append(observation)
            self.actions.append(action)
            self.states.append(state)
            self.clean_observations.append(
                self.simulator.system.get_clean_observation(state)
            )

        total_obj = self.compute_total_objective(
            observations=self.clean_observations,
            actions=self.actions,
        )
        
        print(f'Total objective: {total_obj:.5f}')
        
    
    def plot_observations(
        self,
        observations:pd.DataFrame,
        clean_observations:pd.DataFrame = None,
        y_labels:tuple = (
            r"$x^\mathrm{rel}_\mathrm{jet}$",
            r"$v^\mathrm{rel}_\mathrm{jet}$"
        ),
        title:str = "Observations"
    ):

        fig, axes = plt.subplots(len(y_labels), 1)
        
        for i, label in enumerate(y_labels):
            axes[i].set_ylabel(label)
            
            obs_linestyle = '-'
            if clean_observations is not None:
                clean_observations.iloc[:,i].plot(
                    label='clean observations',
                    ax=axes[i],
                    grid=True,
                    marker='.',
                    linestyle="-"
                )
                obs_linestyle = ''
            
            observations.iloc[:, i].plot(
                label='observations',
                ax=axes[i],
                grid=True,
                marker='x',
                linestyle=obs_linestyle,
                legend=False,
            )
            
        if clean_observations is not None:
            axes[0].legend()

        axes[0].set_xticklabels([])
        
        axes[-1].set_xlabel(r"step number $(t)$")
        fig.suptitle(title)
                
        fig = axes[0].get_figure()
        fig.tight_layout()
        
        return fig
    
    
    def plot_actions(
        self,
        pulse_actions:pd.DataFrame,
        throttle_state:pd.DataFrame = None,
        y_label:str = r"$x^\mathrm{act}_\mathrm{th}$ [µm]",
        title:str = 'Throttle position (action)',
    ):
        fig, ax = plt.subplots()
        
        pulse_actions.plot.line(
            x='step',
            y='action',
            xlabel="step number $(t)$",
            title=title,
            label='action',
            grid=True,
            legend=False,
            ax=ax,
        )
        ax.set_ylabel(y_label)
        
        if throttle_state is not None:
            ax.plot(
                throttle_state,
                marker='.',
                label=r"$x_\mathrm{th}$"
            )
            ax.legend()
            
        fig.tight_layout()
        
        return fig
    
    
    def plot_data(
        self, 
        log_data=True,
    ):
        """Plot results"""
        
        observations = pd.DataFrame(
            data=np.array(self.observations)
        )
        
        clean_observations = pd.DataFrame(
            data=np.array(self.clean_observations)
        )

        obs_fig = self.plot_observations(observations, clean_observations)
        
        actions_arr = self.get_real_actions(self.actions)

        pulse_actions = pd.DataFrame(
            data=actions_arr,
            columns=['step', 'action']
        )
        
        throttle_state = np.array(self.states)[:, 2]
        throttle_state = pd.DataFrame(
            data=throttle_state,
        )

        act_fig = self.plot_actions(pulse_actions, throttle_state)

        if log_data:
            if self.data_path is None:
                now = datetime.now()
                dt_string = now.strftime("%Y-%m-%d_%H%M%S")
                experiment_name = dt_string
                if self.seed is not None:
                    experiment_name += f'_seed_{self.seed}'
                self.data_path = os.path.join(self.root_data_path, experiment_name)
                os.mkdir(self.data_path)
                print("log date and time =", self.data_path)
            
            obs_fig.savefig(
                os.path.join(self.data_path, "observations.pdf"), 
                dpi=self.dpi,
            )
            observations.to_csv(
                os.path.join(self.data_path, "observations.csv")
            )
            clean_observations.to_csv(
                os.path.join(self.data_path, "clean_observations.csv")
            )

            act_fig.savefig(
                os.path.join(self.data_path, "actions.pdf"), 
                dpi=self.dpi
            )
            pulse_actions.to_csv(
                os.path.join(self.data_path, "actions.csv")
            )
            
            # self.data_path = None
            
        plt.show()



class MonteCarloSimulationScenario(SimulationScenario):
    """Run whole REINFORCE procedure"""

    def __init__(
        self,
        N_episodes: int,
        N_iterations: int,
        *args,
        termination_criterion: Callable[
            [np.array, np.array, float, float], bool
        ] = lambda *args: False,
        dt_string: str = None,
        log_each_iteration: bool = True,
        **kwargs,
    ):
        """Initialize scenario for main loop

        Args:
            simulator (Simulator): simulator for computing system dynamics
            policy (PolicyREINFORCE): REINFORCE gradient stepper
            N_episodes (int): number of episodes in one iteration
            N_iterations (int): number of iterations
            discount_factor (float, optional): discount factor for running objectives. Defaults to 1
            termination_criterion (Callable[[np.array, np.array, float, float], bool], optional): criterion for episode termination. Takes observation, action, running_objective, total_objectove. Defaults to lambda*args:False
        """

        super().__init__(*args, **kwargs)
        
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.termination_criterion = termination_criterion

        self.total_objectives_episodic = []
        self.learning_curve = []
        self.last_observations = None
        
        self.iteration_data = []
        
        self.dt_string = dt_string
        self.log_each_iteration = log_each_iteration
    
    
    def run(self) -> None:
        """Run main loop"""

        eps = 0.1 # to calculate first changes
        means_total_objectives = [eps]
        for iteration_idx in range(self.N_iterations):
            if iteration_idx % 10 == 0:
                clear_output(wait=True)
                # WRITE CURRENT RESULT
            for episode_idx in tqdm(range(self.N_episodes)):
                terminated = False
                self.clean_data() # Clean data from previous episode
                # Conduct simulations for one episode
                while self.simulator.step():
                    (
                        step_idx,
                        state,
                        observation,
                        action, # we do not use this
                    ) = self.simulator.get_sim_step_data()

                    new_action = (
                        self.policy.model.sample(torch.tensor(observation).float())
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    discounted_running_objective = self.discount_factor ** (
                        step_idx
                    ) * self.compute_running_objective(observation, new_action)
                    # for learning curve plotting
                    self.total_objective += discounted_running_objective

                    if not terminated and self.termination_criterion(
                        observation,
                        new_action,
                        discounted_running_objective,
                        self.total_objective,
                    ):
                        terminated = True

                    # Thus, if terminated - stop to add data in buffer
                    if not terminated:
                        self.policy.buffer.add_step_data(
                            np.copy(observation),
                            np.copy(new_action),
                            np.copy(discounted_running_objective),
                            step_idx,
                            episode_idx,
                        )
                        # Keep observations and actions
                        self.observations.append(observation)
                        self.actions.append(action)
                        # Keep states and observations without noise
                        self.states.append(state)
                        self.clean_observations.append(
                            self.simulator.system.get_clean_observation(state)
                        )
                        
                    self.simulator.set_action(new_action) # set action for the next step
                
                self.simulator.reset() # before next episode
                
                # Used for learning curve plotting
                # Save for clean observations!
                # REAL PERFORMANCE
                real_total_objective = self.compute_total_objective(
                    observations=self.clean_observations,
                    actions=self.actions,
                )
                
                self.total_objectives_episodic.append(real_total_objective)
                
                # self.total_objective = 0
            
            # Get data for progress estimation
            self.learning_curve.append(np.mean(self.total_objectives_episodic))
            self.last_observations = pd.DataFrame(
                index=self.policy.buffer.episode_ids,
                data=self.policy.buffer.observations.copy(),
            )
            self.last_actions = pd.DataFrame(
                index=self.policy.buffer.episode_ids,
                data=self.policy.buffer.actions.copy(),
            )
            
            self.policy.REINFORCE_step()

            # Output current result of the iteration
            means_total_objectives.append(np.mean(self.total_objectives_episodic)) # means by episodes
            change = (means_total_objectives[-1] / means_total_objectives[-2] - 1) * 100
            sign = "-" if np.sign(change) == -1 else "+"
            print(
                f"Iteration: {iteration_idx + 1} / {self.N_iterations}, "
                + f"mean total cost {round(means_total_objectives[-1], 2)}, "
                + f"% change: {sign}{abs(round(change,2))}, "
                + f"last observation: {self.last_observations.iloc[-1].values.reshape(-1)}",
                end="\n",
            )

            self.total_objectives_episodic = [] # before next iteration
            
            # Save last observations and actions
            if iteration_idx % 1 == 0: # Return 10, if required
                last_observations, last_actions = (
                    self.get_last_observations_and_actions()
                )
                self.iteration_data.append(
                    {
                        'iteration_idx': iteration_idx, 
                        'observations': last_observations, 
                        'actions': last_actions,
                        'states': np.array(self.states),
                        'clean_observations': np.array(self.clean_observations),
                    }
                )
                if self.log_each_iteration:
                    self.log_data()
                

    def get_last_observations_and_actions(self):
        last_episode_index = self.last_observations.index[-1] # self.N_episodes - 1
        last_observations = (
            self.last_observations.loc[last_episode_index].values
        )
        last_actions = self.last_actions.loc[last_episode_index].values
        
        return last_observations, last_actions
    
    
    def plot_learning_curve(
        self, 
        data:pd.DataFrame, 
        y_log_scale=False,
    ):
        if data.shape[0] == 0:
            print('no data to plot')
            return
        na_mask = data.isna()
        not_na_mask = ~na_mask
        interpolated_values = data.interpolate()
        interpolated_values[not_na_mask] = None
        
        fig, ax = plt.subplots()
        
        if y_log_scale:
            ax.set_yscale('log')
        
        data.plot(marker="o", markersize=3, ax=ax)
        interpolated_values.plot(linestyle="--", ax=ax)
        ax.set_title('Total cost by iteration')
        ax.set_xlabel(r'iteration number $(i)$')
        ax.set_ylabel('total cost')
        
        fig.tight_layout()
        
        return fig
    
    
    def log_data(self):
        if self.data_path is None:
            if self.dt_string is None:
                now = datetime.now()
                self.dt_string = now.strftime("%Y-%m-%d_%H%M%S")
            experiment_name = self.dt_string
            if self.seed is not None:
                experiment_name += f'_seed_{self.seed}'
            self.data_path = os.path.join(self.root_data_path, experiment_name)
            os.mkdir(self.data_path)
            print("log folder:", self.data_path)
        
        data = pd.Series(
            index=range(1, len(self.learning_curve) + 1), 
            data=self.learning_curve
        )
        data.to_csv(
            os.path.join(self.data_path, "learning curve.csv")
        )
        
        # Save iteration observations and actions
        with open(
            os.path.join(self.data_path, "iteration_data.pkl"), 'wb'
        ) as file:
            pickle.dump(self.iteration_data, file)
        
        observations = pd.DataFrame(
            data=np.array(self.observations)
        )
        observations.to_csv(
            os.path.join(self.data_path, "observations.csv")
        )
        
        clean_observations = pd.DataFrame(
            data=np.array(self.clean_observations)
        )
        clean_observations.to_csv(
            os.path.join(self.data_path, "clean_observations.csv")
        )
        
        actions_arr = self.get_real_actions(self.actions)
        pulse_actions = pd.DataFrame(
            data=actions_arr,
            columns=['step', 'action']
        )
        pulse_actions.to_csv(
            os.path.join(self.data_path, "actions.csv")
        )
    
    
    def plot_data(self, log_data=True, y_log_scale=False):
        experiment_name = None
        if log_data:
            if self.data_path is None:
                now = datetime.now()
                dt_string = now.strftime("%Y-%m-%d_%H%M%S")
                experiment_name = dt_string
                if self.seed is not None:
                    experiment_name += f'_seed_{self.seed}'
                self.data_path = os.path.join(self.root_data_path, experiment_name)
                os.mkdir(self.data_path)
                print("log date and time =", self.data_path)
        
        # Plot learning curve
        data = pd.Series(
            index=range(1, len(self.learning_curve) + 1), 
            data=self.learning_curve
        )
        
        curve_fig = self.plot_learning_curve(data, y_log_scale=y_log_scale)
        
        # Save learning curve and all iterations
        if log_data:
            curve_fig.savefig(
                os.path.join(self.data_path, "learning_curve.pdf"),
                dpi=self.dpi,
            )
            data.to_csv(
                os.path.join(self.data_path, "learning curve.csv")
            )
            
            # Save iteration observations and actions
            with open(
                os.path.join(self.data_path, "iteration_data.pkl"), 'wb'
            ) as file:
                pickle.dump(self.iteration_data, file)
        
        # Plot and save observations and actions. THEN, REMOVE data_path folder
        super().plot_data(log_data)
        
        plt.show()
        
        