import numpy as np
import pandas as pd

import torch
# Dataset - to create own dataset, DataLoader - for batch generation
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from typing import Tuple, Dict, Optional, Callable, Type, Any



class PDController:
    def __init__(
        self,
        P_coef:float,
        D_coef:float,
    ):
        self.P_coef = P_coef
        self.D_coef = D_coef
        
        
    def get_action(self, observation):
        action = self.P_coef * (1 - observation[0]) - self.D_coef * observation[1]
        action = np.array([action])
        
        return action
    


class IterationBuffer(Dataset):
    """Buffer for experience replay.
    Let us save all our observations and actions for gradient descent step
    
    """

    def __init__(self) -> None:
        """Initialize `IterationBuffer`"""

        super().__init__()
        self.next_baselines = None
        self.nullify_buffer()


    def nullify_buffer(self) -> None:
        """Clear all buffer data"""

        self.episode_ids = []
        self.observations = []
        self.actions = []
        self.running_objectives = []
        self.step_ids = []
        self.total_objectives = None
        self.baselines = None
    
      
    def add_step_data(
        self,
        observation: np.array,
        action: np.array,
        running_objective: float,
        step_id: int,
        episode_id: int,
    ):
        """Add step data to experience replay

        Args:
            observation (np.array): current observation
            action (np.array): current action
            running_objective (float): current running objective
            step_id (int): current step
            episode_id (int): current episode
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.running_objectives.append(running_objective)
        self.episode_ids.append(int(episode_id))
        self.step_ids.append(step_id)
        
    
    def get_N_episodes(self) -> int:
        """Get number of episodes

        Returns:
            int: number of episodes
        """
        return len(np.unique(self.episode_ids))
    
    
    def calculate_tail_total_objectives_and_next_baselines(
        self,
    ) -> Tuple[np.array, float, float]:
        """Calculate tail total costs and baseline. Applied in 'getitem'

        Returns:
            Tuple[np.array, float, float]: tuple of 3 elements # 2 elements, without gradent_normalization_constant
            tail_total_objectives, baseline, gradent_normalization_constant
        """

        unique_episode_ids = np.unique(self.episode_ids)
        # We will keep the same episode indexes in pd.Series for a convenient calculation of the tail total objectives
        running_objectives_series = pd.Series(
            index=self.episode_ids, data=self.running_objectives
        )
        
        # Sum of inverted rows in one episode for all episodes (like summation from the end)
        # Then - invert to get tail sums for each element!
        tail_total_objectives = pd.concat(
            [
                running_objectives_series.loc[i][::-1].cumsum()[::-1]
                for i in unique_episode_ids
            ]
        ).values.reshape(-1)

        # already gothern tail sums for each episode
        # Thus, we need to get an average of tail sums on all episodes for each step 
        next_baselines = (
            pd.Series(index=self.step_ids, data=tail_total_objectives)
            .groupby(level=0) # group by indexes
            .mean()
            .loc[self.step_ids] # expand means on steps of all episodes
            .values.reshape(-1)
        )

        return tail_total_objectives, next_baselines


    def __len__(self) -> int:
        """Get length of buffer. The method should be overrided due to inheritance from `torch.utils.data.Dataset`

        Returns:
            int: length of buffer
        """
        return len(self.observations)


    def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
        """Get item with id `idx`. The method should be overrided due to inheritance from `torch.utils.data.Dataset`

        Args:
            idx (int): id of dataset item to return

        Returns:
            Dict[str, torch.tensor]: dataset item, containing catted observation-action, tail total objective and baselines
        """
        
        # If total_objectives are not filled out, fill out them and next_baselines
        if self.total_objectives is None:
            # Take baseline from next_baseline if the latter exists
            self.baselines = (
                self.next_baselines
                if self.next_baselines is not None
                else np.zeros(shape=len(self.observations))
            )

            (
                self.total_objectives,
                self.next_baselines,
            ) = self.calculate_tail_total_objectives_and_next_baselines()

        observation = torch.tensor(self.observations[idx])
        action = torch.tensor(self.actions[idx])

        return {
            "observations_actions": torch.cat([observation, action]).float(),
            "tail_total_objectives": torch.tensor(self.total_objectives[idx]).float(),
            "baselines": torch.tensor(self.baselines[idx]).float(),
        }


    @property
    def data(self) -> pd.DataFrame:
        """Return current buffer content in pandas.DataFrame

        Returns:
            pd.DataFrame: current buffer content
        """

        return pd.DataFrame(
            {
                "episode_id": self.episode_ids,
                "step_id": self.step_ids,
                "observation": self.observations,
                "action": self.actions,
                "running_objective": self.running_objectives,
            }
        )



class GaussianPDFModel(nn.Module):
    """Model for REINFORCE algorithm that acts like f(x) + normally distributed noise"""

    def __init__(
        self,
        dim_observation: int,
        dim_action: int,
        dim_hidden: int,
        n_hidden_layers: int,
        std: float,
        action_bounds: np.array,
        scale_factor: float,
        leakyrelu_coef=0.2,
    ):
        """Initialize model.

        Args:
            dim_observation (int): dimensionality of observation
            dim_action (int): dimensionality of action
            dim_hidden (int): dimensionality of hidden layer of perceptron
            n_hidden_layers (int): number of hidden layers
            std (float): standard deviation of noise (\\sigma)
            action_bounds (np.array): action bounds with shape (dim_action, 2). `action_bounds[:, 0]` - minimal actions, `action_bounds[:, 1]` - maximal actions
            scale_factor (float): scale factor for last activation (L coefficient) (see details above)
            leakyrelu_coef (float): coefficient for leakyrelu
        """
        super().__init__()

        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden
        self.n_hidden_layers = n_hidden_layers
        self.leakyrelu_coef = leakyrelu_coef
        self.std = std

        self.scale_factor = scale_factor
        self.register_parameter(
            name="scale_tril_matrix",
            param=torch.nn.Parameter(
                (self.std * torch.eye(self.dim_action)).float(),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            name="action_bounds",
            param=torch.nn.Parameter(
                torch.tensor(action_bounds).float(),
                requires_grad=False,
            ),
        )
        
        hidden_list = [
            [
                nn.Linear(self.dim_hidden, self.dim_hidden),
                nn.LeakyReLU(self.leakyrelu_coef),
            ]
            for _ in range(self.n_hidden_layers)
        ]
        
        # is Equivalent of:
        # flat_list = []
        # for sublist in l:
        #     for item in sublist:
        #         flat_list.append(item)
        hidden_list_flat = [item for sublist in hidden_list for item in sublist]

        self.mu_nn = nn.Sequential(
            # Input layer
            nn.Linear(self.dim_observation, self.dim_hidden),
            nn.LeakyReLU(self.leakyrelu_coef),
            # Hiden layers
            *hidden_list_flat,
            # Output layer
            nn.Linear(self.dim_hidden, self.dim_action),
        )
        # init last activation layer
        self.tanh = nn.Tanh()


    def get_unscale_coefs_from_minus_one_one_to_action_bounds(
        self,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Calculate coefficients for linear transformation from [-1, 1] to [U_min, U_max].

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: coefficients
        """

        action_bounds = self.get_parameter("action_bounds")
        
        u_min = self.action_bounds[:,0]
        u_max = self.action_bounds[:,1]
        
        beta_ = (u_min + u_max)/2
        lambda_ = (u_max - u_min)/2
        
        return beta_, lambda_


    def unscale_from_minus_one_one_to_action_bounds(
        self, x: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Linear transformation from [-1, 1] to [U_min, U_max].

        Args:
            x (torch.FloatTensor): tensor to transform

        Returns:
            torch.FloatTensor: transformed tensor
        """
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()

        return x * unscale_multiplier + unscale_bias


    def scale_from_action_bounds_to_minus_one_one(
        self, y: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Linear transformation from [U_min, U_max] to [-1, 1].

        Args:
            y (torch.FloatTensor): tensor to transform

        Returns:
            torch.FloatTensor: transformed tensor
        """
        (
            unscale_bias,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()

        return (y - unscale_bias) / unscale_multiplier


    def get_means(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        """Return mean for MultivariateNormal from `observations`
        \mu_{\theta}(y) \in [-1 + 3\sigma, 1 - 3\sigma]

        Args:
            observations (torch.FloatTensor): observations

        Returns:
            torch.FloatTensor: means
        """
        # First - make forward step with current observations
        nn_result = self.mu_nn(observations)
        # Then, divide by scale_factor (L) and put into the nn.Tanh()
        mu_activation = self.tanh(nn_result/self.scale_factor)
        
        # \\mu_theta(observations)
        # multiply by (1 - 3*std)
        return (1 - 3*self.std)*mu_activation


    def split_to_observations_actions(
        self, observations_actions: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Split input tensor to tuple of observation(s) and action(s)

        Args:
            observations_actions (torch.FloatTensor): tensor of catted observations actions to split

        Raises:
            ValueError: in case if `observations_actions` has dimensinality greater than 2

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: tuple of observation(s) and action(s)
        """
        if len(observations_actions.shape) == 1:
            observation, action = (
                observations_actions[: self.dim_observation],
                observations_actions[self.dim_observation :],
            )
        elif len(observations_actions.shape) == 2:
            observation, action = (
                observations_actions[:, : self.dim_observation],
                observations_actions[:, self.dim_observation :],
            )
        else:
            raise ValueError("Input tensor has unexpected dims")

        return observation, action


    def get_unscale_mean_and_variance(
        self, 
        observations: torch.FloatTensor,
        scale_tril_matrix: torch.nn.parameter.Parameter,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """ Get unscaled mean and covariance matrix for the pdf_Normal

        Args:
            observations (torch.FloatTensor): observations batch
            scale_tril_matrix (torch.nn.parameter.Parameter): covariance matrix

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Unscaled mean and covariance matrix for the pdf_Normal
        """
        # Get means in range [-1, 1]
        mu_scaled = self.get_means(observations)

        # Return back to the action range [U_min, U_max]
        mu_unscaled = self.unscale_from_minus_one_one_to_action_bounds(mu_scaled)
        # Get lambda
        (
            _,
            unscale_multiplier,
        ) = self.get_unscale_coefs_from_minus_one_one_to_action_bounds()
        
        # Get unscaled lower-triangular factor of covariance
        tril_unscaled = scale_tril_matrix * torch.diag(unscale_multiplier)
        
        return mu_unscaled, tril_unscaled
        

    def log_probs(self, batch_of_observations_actions: torch.FloatTensor) -> torch.FloatTensor:
        """Get log pdf from the batch of observations actions

        Args:
            batch_of_observations_actions (torch.FloatTensor): batch of catted observations and actions

        Returns:
            torch.FloatTensor: log pdf(action | observation) for the batch of observations and actions
        """
        observations, actions = self.split_to_observations_actions(
            batch_of_observations_actions
        )

        scale_tril_matrix = self.get_parameter("scale_tril_matrix")
        
        # Get unscaled mean and variance
        (
            mu_unscaled, 
            tril_unscaled
        ) = self.get_unscale_mean_and_variance(observations, scale_tril_matrix)
        
        # Get set of pdfs:
        # pdf_Normal(\\lambda \\mu_theta(observations) + \\beta, \\lambda ** 2 \\sigma ** 2)(actions)
        multi_norm = MultivariateNormal(
            loc=mu_unscaled, # \\lambda \\mu_theta(observations) + \\beta
            scale_tril=tril_unscaled, # \\lambda \\sigma
        )

        # Calculate log pdf(action | observation)
        return multi_norm.log_prob(actions)


    def sample(self, observation: torch.FloatTensor) -> torch.FloatTensor:
        """Sample action from `MultivariteNormal(lambda * self.get_means(observation) + beta, lambda ** 2 * Diag[self.std] ** 2)`

        Args:
            observation (torch.FloatTensor): current observation

        Returns:
            torch.FloatTensor: sampled action
        """
        action_bounds = self.get_parameter("action_bounds")
        scale_tril_matrix = self.get_parameter("scale_tril_matrix")

        # Get unscaled mean and variance
        (
            mu_unscaled, 
            tril_unscaled
        ) = self.get_unscale_mean_and_variance(observation, scale_tril_matrix)
        
        # Get set of pdfs:
        # pdf_Normal(\\lambda \\mu_theta(observations) + \\beta, \\lambda ** 2 \\sigma ** 2)(actions)
        multi_norm = MultivariateNormal(
            loc=mu_unscaled, # \\lambda \\mu_theta(observations) + \\beta
            scale_tril=tril_unscaled, # \\lambda \\sigma
        )
        
        # Sample action from `MultivariteNormal(lambda * self.get_means(observation) + beta, lambda ** 2 * Diag[self.std] ** 2)
        sampled_action = multi_norm.rsample()

        # Clamp prevents getting actions out of the bounds
        return torch.clamp(
            sampled_action, action_bounds[:, 0], action_bounds[:, 1]
        )



class Optimizer:
    """Does gradient step for optimizing model weights"""

    def __init__(
        self,
        model: nn.Module,
        opt_method: Type[torch.optim.Optimizer],
        opt_options: Dict[str, Any],
        shuffle: bool = False,
    ):
        """Initialize Optimizer

        Args:
            model (nn.Module): model which weights we need to optimize
            opt_method (Type[torch.optim.Optimizer]): method type for optimization. For instance, `opt_method=torch.optim.SGD`
            opt_options (Dict[str, Any]): kwargs dict for opt method
            shuffle (bool, optional): whether to shuffle items in dataset. Defaults to True
        """

        self.opt_method = opt_method
        self.opt_options = opt_options
        self.shuffle = shuffle
        self.model = model
        self.optimizer = self.opt_method(self.model.parameters(), **self.opt_options)


    def optimize(
        self,
        objective: Callable[[torch.tensor], torch.tensor],
        dataset: IterationBuffer,
    ) -> None:
        """Do gradient step.

        Args:
            objective (Callable[[torch.tensor], torch.tensor]): objective to optimize
            dataset (Dataset): data for optmization
        """

        # For loading the batch (with dataset size)
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=self.shuffle,
            batch_size=len(dataset),
        )
        batch_sample = next(iter(dataloader)) # return batch - whole dataset
        self.optimizer.zero_grad()
        objective_value = objective(batch_sample)
        objective_value.backward() # calculate gradients
        self.optimizer.step() # apply gradients to model weights



class PolicyREINFORCE:
    def __init__(
        self, model: nn.Module, optimizer: Optimizer, device: str = "cpu", is_with_baseline: bool = True,
    ) -> None:
        """Initialize policy

        Args:
            model (nn.Module): model to optimize
            optimizer (Optimizer): optimizer for `model` weights optimization
            device (str, optional): device for gradient descent optimization procedure. Defaults to "cpu".
            is_with_baseline (bool, optional): whether to use baseline in objective function.
        """

        self.buffer = IterationBuffer()
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.is_with_baseline = is_with_baseline


    def objective(self, batch: Dict["str", torch.tensor]) -> torch.tensor:
        """This method computes a proxy (surrogate) objective specifically for automatic differentiation since its gradient is exactly as in REINFORCE

        Args:
            batch (torch.tensor): batch with catted observations-actions, total objectives and baselines

        Returns:
            torch.tensor: objective value
        """

        observations_actions = batch["observations_actions"].to(self.device)
        tail_total_objectives = batch["tail_total_objectives"].to(self.device)
        baselines = batch["baselines"].to(self.device)
        N_episodes = self.N_episodes
        
        # Get log probs of policy
        log_probs = self.model.log_probs(observations_actions)
        
        # Return the surrogate objective value
        return 1/N_episodes*((tail_total_objectives - baselines)*log_probs).sum()


    def REINFORCE_step(self) -> None:
        """Do gradient REINFORCE step"""

        self.N_episodes = self.buffer.get_N_episodes() # used for objective calc
        # self.model.to(self.device)
        self.optimizer.optimize(self.objective, self.buffer)
        # self.model.to("cpu")
        self.buffer.nullify_buffer() # prepare buffer to next iteration step