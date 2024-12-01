# Experiment 1: DP vs. SAA Solution Quality Assessment
# Small Scale Implementation

import numpy as np
import pandas as pd
from scipy import stats
import time
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import our custom modules
from data_generator import TestConfiguration, create_test_instance
from dynamic_pricing_algorithms import DynamicProgramming, StochasticApproximation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_experiment1(num_trials: int = 5):
    """
    Run Experiment 1: Solution Quality Assessment comparing SAA with DP.
    
    The experiment uses a small but realistic instance that allows for:
    1. Exact solution via Dynamic Programming
    2. Multiple trials of SAA to assess consistency
    3. Statistical comparison of solution quality
    
    Args:
        num_trials: Number of SAA trials to run for statistical significance
        
    Returns:
        Dictionary containing detailed experimental results
    """
    logger.info("Starting Experiment 1: Solution Quality Assessment")
    
    # Create test instance
    config = TestConfiguration()
    test_params = config.get_config(
        test_type='minimal',
        market_condition='standard',
        discretization='coarse'
    )
    
    # Set parameters for a tractable but meaningful test case
    test_params.update({
        'T': 10,  # 10 booking periods
        'N': 5,   # 5-day service horizon
        'C': 5,   # 5 rooms capacity
        'price_min': 100,  # Reasonable price range
        'price_max': 300,
        'alpha': 0.1,     # Smoothing parameters for SAA
        'beta': 0.1
    })
    
    # Create instance with fixed seed for reproducibility
    instance = create_test_instance(
        demand_scenario='base',
        market_condition='standard',
        test_configuration=test_params,
        seed=42
    )
    
    logger.info("\nTest Instance Configuration:")
    logger.info(f"Booking Horizon (T): {test_params['T']} periods")
    logger.info(f"Service Horizon (N): {test_params['N']} days")
    logger.info(f"Room Capacity (C): {test_params['C']} rooms")
    logger.info(f"Price Range: ${test_params['price_min']} - ${test_params['price_max']}")
    
    # Solve using Dynamic Programming
    logger.info("\nSolving with Dynamic Programming...")
    dp = DynamicProgramming(instance)
    start_time = time.time()
    _, dp_revenue = dp.solve()
    dp_time = time.time() - start_time
    
    # Configure SAA parameters
    learning_params = {
        'eta_0': 0.5,        # Initial learning rate
        'gamma': 0.05,       # Learning rate decay
        'eta_min': 0.001,    # Minimum learning rate
        'max_epochs': 1000,
        'batch_size': 64
    }
    
    # Run multiple SAA trials
    logger.info("\nSolving with Stochastic Approximation...")
    saa_results = []
    
    for trial in range(num_trials):
        logger.info(f"\nSAA Trial {trial + 1}/{num_trials}")
        saa = StochasticApproximation(instance, learning_params)
        prices, revenue, solve_time = saa.solve()
        
        # Evaluate final solution with more samples
        final_revenue = saa.evaluate(prices, num_samples=10000)
        saa_results.append({
            'revenue': final_revenue,
            'time': solve_time
        })
        
        logger.info(f"Trial Revenue: ${final_revenue:.2f}")
        logger.info(f"Trial Solution Time: {solve_time:.2f} seconds")
    
    # Compute SAA statistics
    saa_revenues = [r['revenue'] for r in saa_results]
    saa_times = [r['time'] for r in saa_results]
    
    avg_saa_revenue = np.mean(saa_revenues)
    std_saa_revenue = np.std(saa_revenues)
    avg_saa_time = np.mean(saa_times)
    
    # Calculate optimality gap
    gap_percentage = ((dp_revenue - avg_saa_revenue) / dp_revenue) * 100
    
    # Compute confidence interval for SAA revenue
    confidence_level = 0.95
    degrees_of_freedom = num_trials - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    margin_of_error = t_value * (std_saa_revenue / np.sqrt(num_trials))
    
    # Print comprehensive results
    logger.info("\nExperiment 1 Results Summary:")
    logger.info(f"Dynamic Programming Revenue: ${dp_revenue:.2f}")
    logger.info(f"DP Solution Time: {dp_time:.2f} seconds")
    logger.info(f"\nSAA Average Revenue: ${avg_saa_revenue:.2f} ± ${margin_of_error:.2f}")
    logger.info(f"SAA Revenue Std Dev: ${std_saa_revenue:.2f}")
    logger.info(f"SAA Average Solution Time: {avg_saa_time:.2f} seconds")
    logger.info(f"Optimality Gap: {gap_percentage:.2f}%")
    
    return {
        'dp_revenue': dp_revenue,
        'dp_time': dp_time,
        'saa_revenues': saa_revenues,
        'saa_times': saa_times,
        'avg_saa_revenue': avg_saa_revenue,
        'std_saa_revenue': std_saa_revenue,
        'gap_percentage': gap_percentage,
        'confidence_interval': margin_of_error,
        'instance_params': test_params,
        'learning_params': learning_params
    }

if __name__ == "__main__":
    results = run_experiment1(num_trials=5)