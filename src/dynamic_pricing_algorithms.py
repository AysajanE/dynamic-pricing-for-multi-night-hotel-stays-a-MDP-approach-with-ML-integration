import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

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
    """
    Optimized Dynamic Programming solution leveraging concavity properties
    for hotel revenue optimization with multiple-night stays.
    """
    
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
        
        # Pre-compute class information for efficiency
        self._precompute_class_info()
        
        # Initialize value function and policy
        self.value_function: Dict[DPState, float] = {}
        self.optimal_policy: Dict[DPState, Dict[int, float]] = {}
        
        logger.info(f"Initialized DP solver with {len(self.booking_classes)} booking classes")
        
    def _precompute_class_info(self):
        """Pre-compute booking class information for efficient lookup."""
        self.class_stays = {
            (arr, dep): set(range(arr - 1, dep)) 
            for arr, dep in self.booking_classes
        }
        self.stay_lengths = {
            (arr, dep): dep - arr + 1 
            for arr, dep in self.booking_classes
        }
    def _compute_purchase_probability(self, price: float, epsilon: float) -> float:
        """
        Compute purchase probability using linear demand function.
        F̄_b(p_t^b) = 1 - ε_b * p_t^b
        """
        return max(0.0, min(1.0, 1.0 - epsilon * price))    
    
    def _compute_G_function(self, state: DPState, prices: Dict[int, float]) -> float:
        """
        Compute G(x_t, p_t) according to equation (7) in the formulation.

        G(x_t, p_t) = sum_{b in B} u_t^b(x_t, p_t) * (L^b * p_t^b + V(x_t - e^b, t+1) - V(x_t, t+1))
        """
        G_value = 0.0
        current_probs = self.arrival_probs[state.time]

        for booking_class, arrival_prob in current_probs.items():
            if arrival_prob <= 0:
                continue

            arrival, departure = booking_class
            stay_nights = self.class_stays[booking_class]
            length_of_stay = self.stay_lengths[booking_class]

            # Check capacity availability
            has_capacity = all(state.capacity[i] > 0 for i in stay_nights)

            if has_capacity:
                # Calculate average price for the stay
                stay_prices = [prices[i] for i in stay_nights]
                avg_price = sum(stay_prices) / length_of_stay

                # Compute purchase probability
                eps = self.price_sensitivity[booking_class]
                accept_prob = max(0.0, min(1.0, 1.0 - eps * avg_price))

                if accept_prob > 0:
                    # Create next state with reduced capacity
                    next_capacity = list(state.capacity)
                    for i in stay_nights:
                        next_capacity[i] -= 1
                    next_state = DPState(capacity=tuple(next_capacity), time=state.time + 1)

                    # Compute value difference
                    value_current = self.value_function.get(
                        DPState(capacity=state.capacity, time=state.time + 1), 0.0
                    )
                    value_next = self.value_function.get(next_state, 0.0)

                    # Compute contribution to G
                    immediate_revenue = length_of_stay * avg_price
                    future_value_diff = value_next - value_current

                    G_value += arrival_prob * accept_prob * (immediate_revenue + future_value_diff)

        return G_value

    def _optimize_state_prices(self, state: DPState) -> Tuple[float, Dict[int, float]]:
        """
        Optimize prices for current state using concavity of G function.

        V(x_t, t) = V(x_t, t+1) + max_{p_t} G(x_t, p_t)
        """
        # Initialize with current prices at midpoint
        current_prices = {i: (self.price_min + self.price_max) / 2 
                         for i in range(self.N)}

        # Value function at t+1 for current state
        future_value = self.value_function.get(
            DPState(capacity=state.capacity, time=state.time + 1), 0.0
        )

        # Optimize each day's price separately using concavity
        for day in range(self.N):
            def negative_G(price):
                """Objective function for single-day price optimization."""
                current_prices[day] = price
                return -self._compute_G_function(state, current_prices)

            # Use scipy's minimize_scalar with bounds
            result = minimize_scalar(
                negative_G,
                bounds=(self.price_min, self.price_max),
                method='bounded'
            )

            current_prices[day] = result.x

        # Compute optimal G value with final prices
        optimal_G = self._compute_G_function(state, current_prices)

        # Total value is future value plus optimal G
        total_value = future_value + optimal_G

        return total_value, current_prices
    
    def solve(self) -> Tuple[Dict[Tuple[int, ...], Dict[int, float]], float]:
        """
        Solve the DP problem efficiently using concavity properties.
        
        Returns:
            Tuple of (optimal_policy, optimal_value)
        """
        start_time = time.time()
        logger.info("Starting optimized DP solution")
        
        # Initialize boundary conditions
        self._initialize_boundary_conditions()
        
        # Backward induction with efficient price optimization
        for t in range(self.T, 0, -1):
            period_start = time.time()
            states_processed = 0
            
            for capacity in self._generate_capacity_vectors():
                state = DPState(capacity=capacity, time=t)
                optimal_value, optimal_prices = self._optimize_state_prices(state)
                
                self.value_function[state] = optimal_value
                self.optimal_policy[state] = optimal_prices
                states_processed += 1
            
            period_time = time.time() - period_start
            logger.info(
                f"Processed period {t}: {states_processed} states "
                f"in {period_time:.2f} seconds"
            )
        
        # Extract initial state value and policy
        initial_state = DPState(
            capacity=tuple(self.C for _ in range(self.N)),
            time=1
        )
        optimal_value = self.value_function[initial_state]
        
        total_time = time.time() - start_time
        logger.info(
            f"DP solution completed in {total_time:.2f} seconds. "
            f"Optimal value: {optimal_value:.2f}"
        )
        
        return self.optimal_policy, optimal_value
    
    def evaluate_policy(self, policy: Dict, evaluation_paths: List) -> float:
        """Evaluate DP policy on specific sample paths."""
        total_revenue = 0.0
        
        for path in evaluation_paths:
            path_revenue = 0.0
            capacity = np.full(self.N, self.C, dtype=np.int32)
            
            for t, bt, qt in path:
                if bt is not None:
                    state = DPState(capacity=tuple(capacity), time=t)
                    prices = policy.get(state, {})
                    
                    if not prices:  # Handle case where state not in policy
                        continue
                        
                    stay_nights = self.class_stays[bt]
                    has_capacity = all(capacity[i] >= 1 for i in stay_nights)
                    
                    if has_capacity:
                        stay_prices = [prices[i] for i in stay_nights]
                        avg_price = sum(stay_prices) / len(stay_nights)
                        
                        if qt >= avg_price:  # Customer accepts price
                            path_revenue += sum(stay_prices)
                            for i in stay_nights:
                                capacity[i] -= 1
                                
            total_revenue += path_revenue
            
        return total_revenue / len(evaluation_paths)
    
    def _initialize_boundary_conditions(self):
        """Initialize boundary conditions efficiently."""
        logger.info("Initializing boundary conditions")
        
        # Terminal period conditions
        for capacity in self._generate_capacity_vectors():
            terminal_state = DPState(capacity=capacity, time=self.T + 1)
            self.value_function[terminal_state] = 0.0
            
            # Zero capacity conditions for all periods
            if sum(capacity) == 0:
                for t in range(1, self.T + 2):
                    state = DPState(capacity=capacity, time=t)
                    self.value_function[state] = 0.0
                    
    def _generate_capacity_vectors(self) -> List[Tuple[int, ...]]:
        """
        Generate capacity vectors efficiently using numpy.
        """
        capacities = np.array(
            np.meshgrid(
                *[range(self.C + 1) for _ in range(self.N)]
            )
        ).T.reshape(-1, self.N)
        
        return [tuple(cap) for cap in capacities]

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
        
        # Initialize random number generator for training
        train_seed = instance.get('train_seed', None)  # Allow for reproducible training
        self.rng = np.random.default_rng(train_seed)
        
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
    
    def _generate_sample_path(self, rng: np.random.Generator = None) -> List[Tuple]:
        """
        Generate a sample path of customer arrivals and reservation prices.

        The sample path generation follows the two-step process from the theoretical framework:
        1. For each time period, determine if a customer arrives based on total arrival probability
        2. If an arrival occurs, select the booking class and randomly sample reservation price

        Args:
            rng: Optional numpy random number generator. If not provided, uses the instance's
                 default generator for training. This parameter enables consistent evaluation
                 paths when comparing algorithms.

        Returns:
            List of (time, booking_class, reservation_price) tuples, where booking_class
            and reservation_price are None if no arrival occurs in that time period
        """
        # Use provided RNG or fallback to instance RNG for training
        random_generator = rng if rng is not None else self.rng

        path = []
        for t in range(1, self.params.T + 1):
            # Get arrival probabilities for current time period
            probs = self.arrival_probs[t]
            arrival_prob = random_generator.random()
            total_prob = sum(probs.values())

            if arrival_prob < total_prob:
                # Customer arrives - select booking class
                classes = list(probs.keys())
                probabilities = [probs[c] for c in classes]

                # Normalize probabilities for class selection
                normalized_probabilities = [p/total_prob for p in probabilities]
                bt = classes[random_generator.choice(len(classes), p=np.array(normalized_probabilities))]

                # Generate reservation price based on class-specific epsilon
                eps = self.epsilon[bt]
                u = random_generator.random()
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
            
    def solve(self, memory_tracker=None, convergence_tracker=None) -> Tuple[Dict[int, np.ndarray], float, float]:
        """
        Execute the SAA algorithm with memory and convergence tracking.
        
        Parameters:
        -----------
        memory_tracker : MemoryTracker, optional
            Tracker for monitoring memory usage during optimization
        convergence_tracker : ConvergenceTracker, optional
            Tracker for monitoring convergence metrics
            
        Returns:
        --------
        Tuple[Dict[int, np.ndarray], float, float]
            Final prices, revenue, and computation time
        """
        start_time = time.time()
        
        if memory_tracker:
            memory_tracker.capture_snapshot("optimization_start")
        
        # Initialize convergence monitoring
        revenue_history = []
        gradient_history = []
        
        window_size = 20  # Window for checking convergence
        convergence_tol = 1e-4  # Tolerance for gradient norm
        
        for epoch in range(self.learning_params['max_epochs']):
            epoch_start = time.time()
            
            # Compute current learning rate using simple decay schedule
            learning_rate = max(
                self.learning_params['eta_min'],
                self.learning_params['eta_0'] / (1 + self.learning_params['gamma'] * epoch)
            )
            
            if memory_tracker:
                memory_tracker.capture_snapshot(f"epoch_{epoch}_start")
            
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
            
            # Calculate gradient norm for convergence monitoring
            grad_norm = max(np.linalg.norm(grad) for grad in epoch_gradients.values())
            gradient_history.append(grad_norm)
            revenue_history.append(avg_revenue)
            
            if convergence_tracker:
                epoch_time = time.time() - epoch_start
                convergence_tracker.update(
                    objective_value=avg_revenue,
                    gradient_norm=grad_norm,
                    prices=np.array([p for t in range(1, self.params.T + 1) 
                                   for p in self.prices[t]]),
                    iteration_time=epoch_time
                )
            
            # Update prices
            for t in range(1, self.params.T + 1):
                self.prices[t] += learning_rate * epoch_gradients[t]
                # Project prices to feasible range
                self.prices[t] = np.clip(self.prices[t], 
                                       self.params.price_min,
                                       self.params.price_max)
            
            if memory_tracker:
                memory_tracker.capture_snapshot(f"epoch_{epoch}_end")
            
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
        
        if memory_tracker:
            memory_tracker.capture_snapshot("optimization_end")
        
        return self.prices, final_revenue, solve_time
    
    def evaluate(self, prices: Dict[int, np.ndarray], evaluation_paths: List = None) -> float:
        """
        Evaluate pricing policy using provided sample paths or generate new ones.
        
        Args:
            prices: Dictionary mapping time periods to price vectors
            evaluation_paths: Optional list of pre-generated sample paths
            
        Returns:
            Average revenue across sample paths
        """
        if evaluation_paths is None:
            evaluation_paths = [self._generate_sample_path() 
                              for _ in range(1000)]  # Default to 1000 paths
            
        total_revenue = 0.0
        
        for path in evaluation_paths:
            path_revenue = 0.0
            capacity = np.full(self.params.N, self.params.C, dtype=np.int32)
            
            for t, bt, qt in path:
                if bt is not None:
                    stay_nights = self.class_stays[bt]
                    Lb = self.stay_lengths[bt]
                    
                    has_capacity = all(capacity[i] >= 1 for i in stay_nights)
                    
                    if has_capacity:
                        stay_prices = [prices[t][i] for i in stay_nights]
                        price_bt = sum(stay_prices) / Lb
                        
                        if qt >= price_bt:
                            revenue = Lb * price_bt
                            path_revenue += revenue
                            
                            for i in stay_nights:
                                capacity[i] -= 1
            
            total_revenue += path_revenue
        
        return total_revenue / len(evaluation_paths)