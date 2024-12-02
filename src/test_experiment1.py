"""
test_experiment1.py
Unit tests for Experiment 1: Solution Quality Assessment
"""

import unittest
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

from experiment1_solution_quality_assessment import Experiment1Runner
from data_generator import TestConfiguration, create_test_instance
from dynamic_pricing_algorithms import DynamicProgramming, StochasticApproximation

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestExperiment1(unittest.TestCase):
    """Test cases for Experiment 1 implementation."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.runner = Experiment1Runner(
            base_results_dir="../experiments/experiment1_test"
        )
        
        # Create a small test instance
        self.test_instance = self.runner.generate_test_instance(
            capacity=3,
            demand_scenario='base',
            market_condition='standard',
            seed=42
        )
    
    def test_evaluation_consistency(self):
        """Verify that evaluation paths are consistent between algorithms."""
        # Run multiple times with same eval_seed
        eval_seed = 12345
        results = []
        for _ in range(3):
            result = self.runner.run_single_instance(
                self.test_instance, 
                replication=1, 
                eval_seed=eval_seed
            )
            results.append(result)
        
        # Verify results are identical
        for r1, r2 in zip(results[:-1], results[1:]):
            self.assertEqual(r1['dp_revenue'], r2['dp_revenue'], 
                           "DP revenues differ between runs")
            self.assertEqual(r1['saa_revenue'], r2['saa_revenue'], 
                           "SAA revenues differ between runs")
            self.assertEqual(r1['revenue_gap'], r2['revenue_gap'], 
                           "Revenue gaps differ between runs")
            
        logger.info("Evaluation consistency verified")
    
    def test_sample_path_validity(self):
        """Test that generated sample paths are valid for the test instance."""
        # Get a sample result
        result = self.runner.run_single_instance(
            self.test_instance,
            replication=1,
            eval_seed=12345
        )
        
        # Verify result contains expected fields
        self.assertIsNotNone(result)
        expected_fields = ['dp_revenue', 'saa_revenue', 'revenue_gap', 
                          'num_eval_paths', 'eval_seed']
        for field in expected_fields:
            self.assertIn(field, result)
            
        # Verify reasonable revenue values
        self.assertGreater(result['dp_revenue'], 0)
        self.assertGreater(result['saa_revenue'], 0)
        
        logger.info("Sample path validity verified")
    
    def test_seed_reproducibility(self):
        """Test that using the same seed produces identical results."""
        eval_seed = 54321
        
        # Run twice with same seed
        result1 = self.runner.run_single_instance(
            self.test_instance,
            replication=1,
            eval_seed=eval_seed
        )
        
        result2 = self.runner.run_single_instance(
            self.test_instance,
            replication=1,
            eval_seed=eval_seed
        )
        
        # Verify results match exactly
        self.assertEqual(result1['dp_revenue'], result2['dp_revenue'])
        self.assertEqual(result1['saa_revenue'], result2['saa_revenue'])
        self.assertEqual(result1['revenue_gap'], result2['revenue_gap'])
        
        logger.info("Seed reproducibility verified")
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        # Run with different seeds
        result1 = self.runner.run_single_instance(
            self.test_instance,
            replication=1,
            eval_seed=11111
        )
        
        result2 = self.runner.run_single_instance(
            self.test_instance,
            replication=1,
            eval_seed=22222
        )
        
        # Verify results are different (with very high probability)
        self.assertNotEqual(result1['dp_revenue'], result2['dp_revenue'])
        self.assertNotEqual(result1['saa_revenue'], result2['saa_revenue'])
        
        logger.info("Different seeds produce different results - verified")

if __name__ == '__main__':
    unittest.main()