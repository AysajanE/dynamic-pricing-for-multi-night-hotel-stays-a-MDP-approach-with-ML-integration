import itertools
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from dataclasses import dataclass
import time
from data_generator import TestConfiguration, create_test_instance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DPState:
    """Immutable state representation for Dynamic Programming solution."""
    capacity: Tuple[int, ...]  # (capacity_day1, capacity_day2, capacity_day3)
    time: int

class DynamicProgramming:
    """Dynamic Programming solution for hotel revenue optimization."""
    
    def __init__(self, instance: Dict):
        """Initialize using test instance from data_generator."""
        self.params = instance['parameters']
        self.T = self.params.T
        self.N = self.params.N
        self.C = self.params.C
        self.price_min = self.params.price_min
        self.price_max = self.params.price_max
        
        self.booking_classes = instance['booking_classes']
        self.arrival_probs = instance['arrival_probabilities']
        self.price_sensitivity = instance['reservation_price_params']
        
        # Define number of price levels (can be adjusted based on problem size)
        self.num_price_levels = 3  # Using 3 price levels as an example
        
        # Pre-compute price levels for each day
        self.price_levels = np.linspace(self.price_min, self.price_max, self.num_price_levels)
        
        # Generate all possible price combinations for N days
        self.price_combinations = list(itertools.product(self.price_levels, repeat=self.N))
        logger.info(f"Generated {len(self.price_combinations)} price combinations")
        
        # # Use exactly 3 price levels as specified
        # self.price_levels = [self.price_min, (self.price_min + self.price_max)/2, self.price_max]
        # self.price_combinations = list(itertools.product(self.price_levels, repeat=self.N))
        
        # Pre-compute stay patterns
        self.class_stays = {
            (arr, dep): set(range(arr - 1, dep)) 
            for arr, dep in self.booking_classes
        }
        self.stay_lengths = {
            (arr, dep): dep - arr + 1 
            for arr, dep in self.booking_classes
        }
        
        # Initialize value function and policy
        self.value_function: Dict[DPState, float] = {}
        self.optimal_policy: Dict[DPState, Dict[int, float]] = {}
        
    def solve(self) -> Tuple[Dict[Tuple[int, ...], Dict[int, float]], float]:
        """Solve the DP problem and return value functions and optimal policy."""
        logger.info("Starting DP solution")
        
        # Initialize boundary conditions
        self._initialize_boundary_conditions()
        
        # Backward induction
        for t in range(self.T, 0, -1):
            logger.info(f"Processing time period {t}")
            for capacity in self._generate_capacity_vectors():
                state = DPState(capacity=capacity, time=t)
                optimal_value, optimal_prices = self._compute_optimal_decision(state)
                self.value_function[state] = optimal_value
                
                # Store policy as dictionary mapping day to price
                self.optimal_policy[state] = {
                    i+1: price for i, price in enumerate(optimal_prices)
                }
        
        # Extract and format results for capacity = 5
        results = {}
        for state, prices in self.optimal_policy.items():
            if state.capacity[0] == 5:  # Only for first day capacity = 5
                results[state.capacity] = prices
        
        # Get optimal value for initial state
        initial_state = DPState(
            capacity=tuple(self.C for _ in range(self.N)),
            time=1
        )
        optimal_value = self.value_function[initial_state]
        
        return results, optimal_value
    
    def _initialize_boundary_conditions(self):
        """Initialize boundary conditions."""
        for capacity in self._generate_capacity_vectors():
            # Terminal period conditions
            terminal_state = DPState(capacity=capacity, time=self.T + 1)
            self.value_function[terminal_state] = 0.0
            
            # Zero capacity conditions for all periods
            if sum(capacity) == 0:
                for t in range(1, self.T + 2):
                    state = DPState(capacity=capacity, time=t)
                    self.value_function[state] = 0.0
    
    def _compute_optimal_decision(self, state: DPState) -> Tuple[float, Tuple[float, ...]]:
        """Compute optimal value and prices for a state."""
        max_value = float('-inf')
        optimal_prices = self.price_levels[0:self.N]  # Default to minimum prices
        
        for prices in self.price_combinations:
            value = self._compute_expected_value(state, prices)
            if value > max_value:
                max_value = value
                optimal_prices = prices
        
        return max_value, optimal_prices
    
    def _compute_expected_value(self, state: DPState, prices: Tuple[float, ...]) -> float:
        """Compute expected value for state-prices pair."""
        value = 0.0
        current_probs = self.arrival_probs[state.time]
        
        # No arrival case
        no_arrival_prob = 1.0 - sum(current_probs.values())
        if no_arrival_prob > 0:
            next_state = DPState(capacity=state.capacity, time=state.time + 1)
            value += no_arrival_prob * self.value_function[next_state]
        
        # For each possible booking request
        for (arrival, departure), arrival_prob in current_probs.items():
            if arrival_prob <= 0:
                continue
                
            stay_nights = self.class_stays[(arrival, departure)]
            has_capacity = all(state.capacity[day] > 0 for day in stay_nights)
            
            if has_capacity:
                # Calculate average price and acceptance probability
                stay_prices = [prices[day] for day in stay_nights]
                avg_price = sum(stay_prices) / self.stay_lengths[(arrival, departure)]
                
                eps = self.price_sensitivity[(arrival, departure)]
                accept_prob = max(0, 1 - eps * avg_price)
                
                if accept_prob > 0:
                    # Acceptance case
                    next_capacity = list(state.capacity)
                    for day in stay_nights:
                        next_capacity[day] -= 1
                    
                    next_state = DPState(capacity=tuple(next_capacity), time=state.time + 1)
                    immediate_revenue = sum(stay_prices)
                    future_value = self.value_function[next_state]
                    
                    value += arrival_prob * accept_prob * (immediate_revenue + future_value)
                
                # Rejection case
                if accept_prob < 1:
                    reject_prob = 1 - accept_prob
                    next_state = DPState(capacity=state.capacity, time=state.time + 1)
                    value += arrival_prob * reject_prob * self.value_function[next_state]
            else:
                # No capacity case
                next_state = DPState(capacity=state.capacity, time=state.time + 1)
                value += arrival_prob * self.value_function[next_state]
        
        return value
    
    def _generate_capacity_vectors(self) -> List[Tuple[int, ...]]:
        """Generate all possible capacity vectors."""
        return [tuple(cap) for cap in itertools.product(range(self.C + 1), repeat=self.N)]

class StochasticApproximation:
    """
    Implementation of the Stochastic Approximation Algorithm for hotel dynamic pricing.
    
    This implementation follows the exact methodology described in the theoretical framework,
    incorporating smoothed decision functions and proper gradient calculations.
    """
    
    def __init__(self, instance: Dict, learning_params: Dict = None):
        """
        Initialize the SAA algorithm with problem instance and learning parameters.
        
        Args:
            instance: Dictionary containing problem parameters and data
            learning_params: Dictionary containing learning rate parameters:
                - eta_0: Initial learning rate
                - gamma: Learning rate decay parameter
                - eta_min: Minimum learning rate
                - max_epochs: Maximum number of training epochs
                - batch_size: Mini-batch size for gradient computation
        """
        # Extract instance parameters
        self.params = instance['parameters']
        self.booking_classes = instance['booking_classes']
        self.arrival_probs = instance['arrival_probabilities']
        self.epsilon = instance['reservation_price_params']
        
        # Set learning parameters
        default_learning_params = {
            'eta_0': 0.1,
            'gamma': 0.1,
            'eta_min': 0.001,
            'max_epochs': 1000,
            'batch_size': 32
        }
        self.learning_params = {**default_learning_params, **(learning_params or {})}
        
        # Initialize smoothing parameters from instance
        self.alpha = self.params.alpha  # Price acceptance smoothing
        self.beta = self.params.beta    # Capacity smoothing
        
        # Initialize prices and precompute class information
        self.prices = self._initialize_prices()
        self._precompute_class_info()
        
        logger.info(f"Initialized SAA with {len(self.booking_classes)} booking classes")
    
    def _initialize_prices(self) -> Dict[int, np.ndarray]:
        """Initialize price vectors for each booking period."""
        return {t: np.full(self.params.N, (self.params.price_min + self.params.price_max) / 2)
                for t in range(1, self.params.T + 1)}
    
    def _precompute_class_info(self):
        """Precompute booking class information for efficient computation."""
        self.class_stays = {}
        self.stay_lengths = {}
        for arrival, departure in self.booking_classes:
            self.class_stays[(arrival, departure)] = list(range(arrival - 1, departure))
            self.stay_lengths[(arrival, departure)] = departure - arrival + 1
    
    def _compute_smoothed_decision(self, 
                                 price_bt: float,
                                 qt: float,
                                 xt: np.ndarray,
                                 stay_nights: List[int]) -> float:
        """
        Compute smoothed decision function value.
        
        Args:
            price_bt: Average price for booking class b at time t
            qt: Customer's reservation price
            xt: Current capacity vector
            stay_nights: List of nights required for the stay
            
        Returns:
            Smoothed decision function value
        """
        # Price acceptance smoothing
        sp = 1 / (1 + np.exp(-self.alpha * (qt - price_bt)))
        
        # Capacity smoothing for each required night
        sx = np.prod([1 / (1 + np.exp(-self.beta * (xt[i] - 1))) for i in stay_nights])
        
        return sp * sx
    
    def _compute_decision_gradients(self,
                               i: int,
                               price_bt: float,
                               qt: float,
                               xt: np.ndarray,
                               stay_nights: List[int],
                               Lb: int) -> Tuple[float, float]:
        """
        Compute gradients of the decision function with respect to price and capacity.
        
        Args:
            i: Index of the day
            price_bt: Average price for booking class b_t
            qt: Customer's reservation price
            xt: Current capacity vector
            stay_nights: List of nights required for the stay
            Lb: Length of stay
            
        Returns:
            Tuple of (∂ũ/∂p_(t,i), ∂ũ/∂x_(t,i))
        """
        if i not in stay_nights:
            return 0.0, 0.0
            
        # Compute common terms
        sp = 1 / (1 + np.exp(-self.alpha * (qt - price_bt)))
        sx_prod = np.prod([1 / (1 + np.exp(-self.beta * (xt[j] - 1))) 
                          for j in stay_nights])
        
        # Compute price gradient
        du_dp = -(self.alpha / Lb) * sp * (1 - sp) * sx_prod
        
        # Compute capacity gradient
        sx_i = 1 / (1 + np.exp(-self.beta * (xt[i] - 1)))
        du_dx = self.beta * sp * (1 - sx_i) * sx_prod
        
        return du_dp, du_dx
    
    def _compute_immediate_revenue(self,
                                 bt: Tuple[int, int],
                                 price_bt: float,
                                 ut: float) -> float:
        """
        Compute immediate revenue for an accepted booking.
        
        Args:
            bt: Booking class tuple (arrival, departure)
            price_bt: Average price for the stay
            ut: Decision function value
            
        Returns:
            Immediate revenue
        """
        Lb = self.stay_lengths[bt]
        return Lb * price_bt * ut
    
    def _compute_revenue_gradients(self,
                               i: int,
                               price_bt: float,
                               qt: float,
                               xt: np.ndarray,
                               stay_nights: List[int],
                               Lb: int) -> Tuple[float, float]:
        """
        Compute gradients of immediate revenue with respect to price and capacity.
        
        For price gradient (i ∈ N^(b_t)):
        ∂R_t/∂p_(t,i) = ũ^(b_t)_t + Lb * p^(b_t)_t * ∂ũ^(b_t)_t/∂p_(t,i)
        
        For capacity gradient (i ∈ N^(b_t)):
        ∂R_t/∂x_(t,i) = Lb * p^(b_t)_t * ∂ũ^(b_t)_t/∂x_(t,i)
        
        Args:
            i: Index of the day
            price_bt: Average price for booking class b_t
            qt: Customer's reservation price
            xt: Current capacity vector
            stay_nights: List of nights required for the stay
            Lb: Length of stay
            
        Returns:
            Tuple of (∂R_t/∂p_(t,i), ∂R_t/∂x_(t,i))
        """
        if i not in stay_nights:
            return 0.0, 0.0
            
        # Compute base decision function value
        ut = self._compute_smoothed_decision(price_bt, qt, xt, stay_nights)
        
        # Compute decision function gradients
        du_dp, du_dx = self._compute_decision_gradients(i, price_bt, qt, xt, stay_nights, Lb)
        
        # Compute revenue gradients
        dR_dp = ut + Lb * price_bt * du_dp  # Price gradient includes both immediate and derivative terms
        dR_dx = Lb * price_bt * du_dx       # Capacity gradient only includes derivative term
        
        return dR_dp, dR_dx
    
    def _generate_sample_path(self) -> List[Tuple]:
        """
        Generate a sample path of customer arrivals and reservation prices.
        
        The sample path generation follows the two-step process from the theoretical framework:
        1. For each time period, determine if a customer arrives based on total arrival probability
        2. If an arrival occurs, select the booking class and randomly sample reservation price
        
        Returns:
            List of (time, booking_class, reservation_price) tuples, where booking_class
            and reservation_price are None if no arrival occurs in that time period
        """
        path = []
        for t in range(1, self.params.T + 1):
            # Get arrival probabilities for current time period
            probs = self.arrival_probs[t]
            arrival_prob = np.random.random()
            total_prob = sum(probs.values())
            
            if arrival_prob < total_prob:
                # Customer arrives - select booking class
                classes = list(probs.keys())
                probabilities = [probs[c] for c in classes]
                
                # Normalize probabilities for class selection
                normalized_probabilities = [p/total_prob for p in probabilities]
                bt = classes[np.random.choice(len(classes), p=np.array(normalized_probabilities))]
                
                # Generate reservation price based on class-specific epsilon
                eps = self.epsilon[bt]
                u = np.random.random()
                qt = u / eps    # Use inverse transform method for CDF: epsilon*p
                
                path.append((t, bt, qt))
            else:
                # No arrival in this time period
                path.append((t, None, None))
        
        return path
    
    def _forward_pass(self, sample_path: List[Tuple]) -> Tuple[Dict, Dict, Dict]:
        """
        Perform forward pass through the sample path following Algorithm Phase I.b.2.
        
        Args:
            sample_path: List of (time, booking_class, reservation_price) tuples
            
        Returns:
            Tuple of (revenues, decision_values, capacities) where each is a dictionary
            mapping time periods to their respective values
        """
        # Initialize storage for algorithm outputs
        revenues = {}          # Store R_t for each t
        decision_values = {}   # Store ũ_t^(b_t) for each t
        capacities = {1: np.full(self.params.N, self.params.C, dtype=np.float64)}  # Initialize x_1 with float type for training
        
        for t, bt, qt in sample_path:
            if bt is not None:
                # Extract booking class information
                stay_nights = self.class_stays[bt]
                Lb = self.stay_lengths[bt]
                
                # Compute average price for the stay
                stay_prices = [self.prices[t][i] for i in stay_nights]
                price_bt = sum(stay_prices) / Lb
                
                # Compute smoothed decision function
                ut = self._compute_smoothed_decision(price_bt, qt, capacities[t], stay_nights)
                
                # Compute immediate revenue
                rt = self._compute_immediate_revenue(bt, price_bt, ut)
                
                # Update capacity
                next_capacity = capacities[t].copy()
                for i in stay_nights:
                    next_capacity[i] -= ut
                
                # Store values
                revenues[t] = rt
                decision_values[t] = ut
                capacities[t + 1] = next_capacity
                
            else:
                # No arrival case
                revenues[t] = 0
                decision_values[t] = 0
                capacities[t + 1] = capacities[t].copy()
        
        return revenues, decision_values, capacities
    
    def _backward_pass(
            self,
            revenues: Dict[int, float],
            decision_values: Dict[int, float],
            capacities: Dict[int, np.ndarray],
            sample_path: List[Tuple]) -> Dict[int, np.ndarray]:
        """
        Perform backward pass to compute gradients following Algorithm Phase I.b.3.
        
        Args:
            revenues: Dictionary mapping t to R_t
            decision_values: Dictionary mapping t to ũ_t^(b_t)
            capacities: Dictionary mapping t to x_t
            sample_path: List of (time, booking_class, reservation_price) tuples
            
        Returns:
            Dictionary mapping time periods to gradient accumulators ∇_p^(t,i)
        """
        # Initialize gradient accumulators and terminal conditions
        gradient_accumulators = {t: np.zeros(self.params.N) 
                               for t in range(1, self.params.T + 1)}
        dJ_dx_next = np.zeros(self.params.N)  # ∂J/∂x_(T+1,i) = 0
        
        # Backward pass from T to 1
        for t in range(self.params.T, 0, -1):
            t_idx = t - 1  # Convert to 0-based index for sample path
            _, bt, qt = sample_path[t_idx]
            
            if bt is not None:
                # Extract booking class information
                stay_nights = self.class_stays[bt]
                Lb = self.stay_lengths[bt]
                
                # Compute average price for the stay
                stay_prices = [self.prices[t][i] for i in stay_nights]
                price_bt = sum(stay_prices) / Lb
                
                # Get stored values
                # ut = decision_values[t]
                xt = capacities[t]
                
                # Initialize gradient arrays for current time period
                dJ_dp = np.zeros(self.params.N)
                dJ_dx = np.zeros(self.params.N)
                
                # Compute gradients for each day
                for i in range(self.params.N):
                    # Compute immediate revenue gradients
                    dR_dp, dR_dx = self._compute_revenue_gradients(
                        i, price_bt, qt, xt, stay_nights, Lb)
                    
                    # Compute decision function gradients
                    du_dp, du_dx = self._compute_decision_gradients(
                        i, price_bt, qt, xt, stay_nights, Lb)
                    
                    # Sum gradient terms for nights in the stay
                    gradient_sum_dp = 0.0
                    gradient_sum_dx = 0.0
                    for k in stay_nights:
                        gradient_sum_dp += du_dp * dJ_dx_next[k]
                        gradient_sum_dx += du_dx * dJ_dx_next[k]
                    
                    # Update total derivatives
                    dJ_dp[i] = dR_dp - gradient_sum_dp
                    dJ_dx[i] = dR_dx + dJ_dx_next[i] - gradient_sum_dx
                
                # Accumulate gradients
                gradient_accumulators[t] += dJ_dp
                
            else:
                # No arrival case - only capacity gradients persist
                dJ_dx = dJ_dx_next.copy()
            
            # Update dJ_dx_next for next iteration
            dJ_dx_next = dJ_dx
            
        return gradient_accumulators
    
    def _update_prices(self, gradients: Dict[int, np.ndarray], learning_rate: float):
        """Update prices using computed gradients."""
        for t in range(1, self.params.T + 1):
            self.prices[t] += learning_rate * gradients[t]
            # Project prices to feasible range
            self.prices[t] = np.clip(self.prices[t], 
                                   self.params.price_min, 
                                   self.params.price_max)
    
#     def solve(self) -> Tuple[Dict[int, np.ndarray], float, float]:
#         """
#         Execute the SAA algorithm with proper gradient-based optimization and convergence checking.
        
#         Implements Phase I.c: Price Update and Convergence Check from the theoretical framework.
        
#         Returns:
#             Tuple of (final_prices, final_revenue, solve_time)
#         """
#         start_time = time.time()
        
#         # Initialize convergence checking
#         prev_gradients = None
#         num_stable_iterations = 0
#         convergence_threshold = self.learning_params.get('convergence_threshold', 1e-6)
#         min_stable_iterations = self.learning_params.get('min_stable_iterations', 5)
        
#         for epoch in range(self.learning_params['max_epochs']):
#             # Initialize gradient accumulators for the epoch
#             epoch_gradients = {t: np.zeros(self.params.N) 
#                              for t in range(1, self.params.T + 1)}
            
#             # Compute current learning rate using decay schedule
#             learning_rate = max(
#                 self.learning_params['eta_min'],
#                 self.learning_params['eta_0'] / (1 + self.learning_params['gamma'] * epoch)
#             )
            
#             # Process mini-batch
#             for _ in range(self.learning_params['batch_size']):
#                 # Generate sample path
#                 sample_path = self._generate_sample_path()
                
#                 # Forward pass through sample path
#                 revenues, decision_values, capacities = self._forward_pass(sample_path)
                
#                 # Backward pass to compute gradients
#                 gradients = self._backward_pass(revenues, decision_values, capacities, sample_path)
                
#                 # Accumulate gradients
#                 for t in range(1, self.params.T + 1):
#                     epoch_gradients[t] += gradients[t]
            
#             # Average gradients over mini-batch
#             for t in range(1, self.params.T + 1):
#                 epoch_gradients[t] /= self.learning_params['batch_size']
            
#             # Check convergence based on gradient stability
#             if prev_gradients is not None:
#                 max_gradient_change = max(
#                     np.max(np.abs(epoch_gradients[t] - prev_gradients[t]))
#                     for t in range(1, self.params.T + 1)
#                 )
                
#                 if max_gradient_change < convergence_threshold:
#                     num_stable_iterations += 1
#                     if num_stable_iterations >= min_stable_iterations:
#                         logger.info(f"Converged after {epoch + 1} epochs")
#                         break
#                 else:
#                     num_stable_iterations = 0
            
#             # Store current gradients for next iteration
#             prev_gradients = {t: np.copy(grad) for t, grad in epoch_gradients.items()}
            
#             # Update prices using averaged gradients
#             for t in range(1, self.params.T + 1):
#                 self.prices[t] += learning_rate * epoch_gradients[t]
#                 # Project prices onto feasible set [price_min, price_max]
#                 self.prices[t] = np.clip(self.prices[t], 
#                                        self.params.price_min,
#                                        self.params.price_max)
            
#             if epoch % 100 == 0:
#                 logger.info(f"Epoch {epoch}: Max Gradient Norm = "
#                           f"{max(np.linalg.norm(grad) for grad in epoch_gradients.values()):.6f}, "
#                           f"Learning Rate = {learning_rate:.6f}")
        
#         solve_time = time.time() - start_time
        
#         # Compute final revenue for reporting
#         final_revenue = self.evaluate(self.prices)
        
#         return self.prices, final_revenue, solve_time

    def solve(self) -> Tuple[Dict[int, np.ndarray], float, float]:
        """
        Execute the SAA algorithm with proper convergence checking.
        
        Uses gradient-based convergence criteria and monitors revenue improvements
        to determine when the algorithm has converged to an optimal solution.
        
        Returns:
            Tuple of (final_prices, final_revenue, solve_time)
        """
        start_time = time.time()
        
        # Initialize convergence monitoring
        revenue_history = []
        gradient_history = []
        window_size = 20  # Window for checking convergence
        convergence_tol = 1e-4  # Tolerance for gradient norm
        
        for epoch in range(self.learning_params['max_epochs']):
            # Compute current learning rate using simple decay schedule
            learning_rate = max(
                self.learning_params['eta_min'],
                self.learning_params['eta_0'] / (1 + self.learning_params['gamma'] * epoch)
            )
            
            # Initialize epoch statistics
            epoch_gradients = {t: np.zeros(self.params.N) 
                             for t in range(1, self.params.T + 1)}
            epoch_revenue = 0.0
            
            # Process mini-batch
            for _ in range(self.learning_params['batch_size']):
                # Generate sample path
                sample_path = self._generate_sample_path()
                
                # Forward pass
                revenues, decision_values, capacities = self._forward_pass(sample_path)
                
                # Backward pass
                gradients = self._backward_pass(revenues, decision_values, capacities, sample_path)
                
                # Accumulate gradients and revenue
                for t in range(1, self.params.T + 1):
                    epoch_gradients[t] += gradients[t]
                epoch_revenue += sum(revenues.values())
            
            # Average gradients and revenue over mini-batch
            for t in range(1, self.params.T + 1):
                epoch_gradients[t] /= self.learning_params['batch_size']
            avg_revenue = epoch_revenue / self.learning_params['batch_size']
            
            # Update prices
            for t in range(1, self.params.T + 1):
                self.prices[t] += learning_rate * epoch_gradients[t]
                # Project prices to feasible range
                self.prices[t] = np.clip(self.prices[t], 
                                       self.params.price_min,
                                       self.params.price_max)
            
            # Calculate gradient norm for convergence check
            grad_norm = max(np.linalg.norm(grad) for grad in epoch_gradients.values())
            gradient_history.append(grad_norm)
            revenue_history.append(avg_revenue)
            
            # Check convergence
            if len(gradient_history) >= window_size:
                # Keep only the last window_size elements
                gradient_history = gradient_history[-window_size:]
                revenue_history = revenue_history[-window_size:]
                
                # Check if gradient norms are consistently small
                if all(norm < convergence_tol for norm in gradient_history):
                    logger.info(f"Converged after {epoch + 1} epochs: "
                              f"gradient norm below tolerance")
                    break
                
                # Check if revenue has stabilized
                revenue_change = abs(revenue_history[-1] - revenue_history[0]) / abs(revenue_history[0])
                if revenue_change < convergence_tol and grad_norm < convergence_tol:
                    logger.info(f"Converged after {epoch + 1} epochs: "
                              f"revenue stabilized and gradient norm small")
                    break
            
            # Log progress every 50 epochs
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Revenue = {avg_revenue:.2f}, "
                          f"Gradient Norm = {grad_norm:.6f}, "
                          f"Learning Rate = {learning_rate:.6f}")
        
        solve_time = time.time() - start_time
        final_revenue = self.evaluate(self.prices)
        
        return self.prices, final_revenue, solve_time
    
    def evaluate(self, prices: Dict[int, np.ndarray], num_samples: int = 1000) -> float:
        """
        Evaluate a pricing policy using Monte Carlo simulation with the original discrete
        decision function (not the smoothed version used in training).
        
        Args:
            prices: Dictionary mapping time periods to price vectors
            num_samples: Number of sample paths to evaluate
            
        Returns:
            Average revenue across sample paths
        """
        total_revenue = 0.0
        
        for _ in range(num_samples):
            # Generate sample path
            sample_path = self._generate_sample_path()
            path_revenue = 0.0
            capacity = np.full(self.params.N, self.params.C, dtype=np.int32)  # Integer capacity
            
            # Process each time period
            for t, bt, qt in sample_path:
                if bt is not None:
                    # Extract booking class information
                    stay_nights = self.class_stays[bt]
                    Lb = self.stay_lengths[bt]
                    
                    # Check capacity (must have at least 1 room for all nights)
                    has_capacity = all(capacity[i] >= 1 for i in stay_nights)
                    
                    if has_capacity:
                        # Compute average price for the stay
                        stay_prices = [prices[t][i] for i in stay_nights]
                        price_bt = sum(stay_prices) / Lb
                        
                        # Check if customer accepts price
                        if qt >= price_bt:
                            # Accept booking
                            revenue = Lb * price_bt
                            path_revenue += revenue
                            
                            # Update capacity (integer updates)
                            for i in stay_nights:
                                capacity[i] -= 1
            
            total_revenue += path_revenue
        
        return total_revenue / num_samples