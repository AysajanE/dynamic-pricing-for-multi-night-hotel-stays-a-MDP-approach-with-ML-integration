import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import itertools

from data_generator import TestConfiguration, create_test_instance
from dynamic_pricing_algorithms import DynamicProgramming, StochasticApproximation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Experiment1Runner:
    """
    Runner class for Experiment 1: Solution Quality Assessment
    Compares SAA performance against optimal DP solution for small instances.
    """
    
    def __init__(self, output_dir: str = "results/experiment1"):
        """Initialize experiment runner with configuration."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define experiment parameters
        self.T = 5  # Booking horizon
        self.N = 3  # Service horizon
        
        # Define parameter ranges for test instances
        self.capacity_levels = [3, 5, 7]  # Small capacities for tractable DP
        self.demand_scenarios = ['low', 'base', 'high']
        self.market_conditions = ['budget', 'standard', 'luxury']
        
        # SAA learning parameters
        self.learning_params = {
            'eta_0': 0.5,        # Initial learning rate
            'gamma': 0.05,       # Learning rate decay
            'eta_min': 0.001,    # Minimum learning rate
            'max_epochs': 1000,  # Maximum training epochs
            'batch_size': 64     # Mini-batch size
        }
        
        # Statistical parameters
        self.num_replications = 30  # Number of replications per configuration
        self.confidence_level = 0.95
        
    def generate_test_instance(self, 
                             capacity: int,
                             demand_scenario: str,
                             market_condition: str,
                             seed: int) -> Dict:
        """Generate a single test instance with specified parameters."""
        # Configure test parameters
        config = TestConfiguration()
        test_params = config.get_config(
            test_type='minimal',
            market_condition=market_condition,
            discretization='standard'
        )
        
        # Override with experiment-specific parameters
        test_params.update({
            'T': self.T,
            'N': self.N,
            'C': capacity
        })
        
        # Create and return test instance
        return create_test_instance(
            demand_scenario=demand_scenario,
            market_condition=market_condition,
            test_configuration=test_params,
            seed=seed
        )
    
    def run_single_instance(self,
                          instance: Dict,
                          replication: int) -> Dict:
        """Run both DP and SAA on a single test instance."""
        try:
            # Solve using Dynamic Programming
            dp = DynamicProgramming(instance)
            dp_start = datetime.now()
            _, dp_revenue = dp.solve()
            dp_time = (datetime.now() - dp_start).total_seconds()
            
            # Solve using SAA
            saa = StochasticApproximation(instance, self.learning_params)
            saa_start = datetime.now()
            prices, saa_revenue, saa_time = saa.solve()
            
            # Compute revenue gap
            revenue_gap = ((dp_revenue - saa_revenue) / dp_revenue) * 100
            
            return {
                'capacity': instance['parameters'].C,
                'demand_scenario': instance['scenario_info']['demand_scenario'],
                'market_condition': instance['scenario_info']['market_condition'],
                'replication': replication,
                'dp_revenue': dp_revenue,
                'dp_time': dp_time,
                'saa_revenue': saa_revenue,
                'saa_time': saa_time,
                'revenue_gap': revenue_gap
            }
            
        except Exception as e:
            logger.error(f"Error processing instance: {str(e)}")
            return None
    
    def run_experiment(self, num_workers: int = 4) -> pd.DataFrame:
        """Run the complete experiment with all parameter combinations."""
        logger.info("Starting Experiment 1: Solution Quality Assessment")
        
        # Generate parameter combinations
        combinations = list(itertools.product(
            self.capacity_levels,
            self.demand_scenarios,
            self.market_conditions,
            range(self.num_replications)
        ))
        
        # Initialize results storage
        results = []
        
        # Run experiments in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_params = {
                executor.submit(
                    self.run_single_instance,
                    self.generate_test_instance(
                        capacity=c,
                        demand_scenario=d,
                        market_condition=m,
                        seed=100*r + 1
                    ),
                    r
                ): (c, d, m, r) for c, d, m, r in combinations
            }
            
            for future in future_to_params:
                result = future.result()
                if result is not None:
                    results.append(result)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save raw results
        results_df.to_csv(self.output_dir / 'raw_results.csv', index=False)
        
        return results_df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """Perform statistical analysis on experiment results."""
        analysis = {}
        
        # Overall statistics
        analysis['overall'] = {
            'mean_revenue_gap': results_df['revenue_gap'].mean(),
            'std_revenue_gap': results_df['revenue_gap'].std(),
            'mean_dp_time': results_df['dp_time'].mean(),
            'mean_saa_time': results_df['saa_time'].mean()
        }
        
        # Paired t-test for revenue differences
        t_stat, p_value = stats.ttest_rel(
            results_df['dp_revenue'],
            results_df['saa_revenue']
        )
        
        analysis['statistical_tests'] = {
            't_statistic': t_stat,
            'p_value': p_value
        }
        
        # Confidence intervals for revenue gap
        ci = stats.t.interval(
            self.confidence_level,
            len(results_df) - 1,
            loc=results_df['revenue_gap'].mean(),
            scale=stats.sem(results_df['revenue_gap'])
        )
        
        analysis['confidence_intervals'] = {
            'revenue_gap_lower': ci[0],
            'revenue_gap_upper': ci[1]
        }
        
        # Analysis by capacity level
        analysis['by_capacity'] = results_df.groupby('capacity').agg({
            'revenue_gap': ['mean', 'std'],
            'dp_time': 'mean',
            'saa_time': 'mean'
        }).to_dict()
        
        # Save analysis results
        with open(self.output_dir / 'analysis_results.json', 'w') as f:
            json.dump(analysis, f, indent=4)
        
        return analysis
    
    def create_visualizations(self, results_df: pd.DataFrame):
        """Create and save visualizations of experimental results."""
        # Set style
        plt.style.use('seaborn')
        
        # 1. Revenue Comparison Bar Chart
        plt.figure(figsize=(12, 6))
        sns.barplot(data=results_df, x='capacity', y='dp_revenue', 
                   hue='demand_scenario', ci=95)
        plt.title('DP Revenue by Capacity and Demand Scenario')
        plt.xlabel('Capacity Level')
        plt.ylabel('Revenue')
        plt.savefig(self.output_dir / 'revenue_comparison.png')
        plt.close()
        
        # 2. Revenue Gap Box Plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results_df, x='capacity', y='revenue_gap',
                   hue='market_condition')
        plt.title('Revenue Gap Distribution by Capacity and Market Condition')
        plt.xlabel('Capacity Level')
        plt.ylabel('Revenue Gap (%)')
        plt.savefig(self.output_dir / 'revenue_gap_distribution.png')
        plt.close()
        
        # 3. Solution Time Comparison
        plt.figure(figsize=(12, 6))
        results_df_melted = pd.melt(results_df, 
                                   id_vars=['capacity'],
                                   value_vars=['dp_time', 'saa_time'],
                                   var_name='Algorithm',
                                   value_name='Time (seconds)')
        sns.boxplot(data=results_df_melted, x='capacity', y='Time (seconds)',
                   hue='Algorithm')
        plt.title('Solution Time Comparison')
        plt.xlabel('Capacity Level')
        plt.savefig(self.output_dir / 'solution_time_comparison.png')
        plt.close()
    
    def generate_report(self, results_df: pd.DataFrame, analysis: Dict):
        """Generate a comprehensive report of experimental results."""
        report = []
        report.append("# Experiment 1: Solution Quality Assessment Report")
        report.append("\n## Overview")
        report.append(f"- Total test instances: {len(results_df)}")
        report.append(f"- Capacity levels: {self.capacity_levels}")
        report.append(f"- Demand scenarios: {self.demand_scenarios}")
        report.append(f"- Market conditions: {self.market_conditions}")
        report.append(f"- Replications per configuration: {self.num_replications}")
        
        report.append("\n## Overall Results")
        report.append(f"- Mean revenue gap: {analysis['overall']['mean_revenue_gap']:.2f}%")
        report.append(f"- Revenue gap 95% CI: [{analysis['confidence_intervals']['revenue_gap_lower']:.2f}%, "
                     f"{analysis['confidence_intervals']['revenue_gap_upper']:.2f}%]")
        report.append(f"- Mean DP solution time: {analysis['overall']['mean_dp_time']:.2f} seconds")
        report.append(f"- Mean SAA solution time: {analysis['overall']['mean_saa_time']:.2f} seconds")
        
        report.append("\n## Statistical Analysis")
        report.append(f"- T-statistic: {analysis['statistical_tests']['t_statistic']:.4f}")
        report.append(f"- P-value: {analysis['statistical_tests']['p_value']:.4f}")
        
        report.append("\n## Results by Capacity Level")
        for capacity in self.capacity_levels:
            cap_data = analysis['by_capacity']
            report.append(f"\nCapacity = {capacity}")
            report.append(f"- Mean revenue gap: {cap_data['revenue_gap']['mean'][capacity]:.2f}%")
            report.append(f"- Revenue gap std: {cap_data['revenue_gap']['std'][capacity]:.2f}%")
            report.append(f"- Mean DP time: {cap_data['dp_time']['mean'][capacity]:.2f} seconds")
            report.append(f"- Mean SAA time: {cap_data['saa_time']['mean'][capacity]:.2f} seconds")
        
        # Save report
        with open(self.output_dir / 'experiment_report.md', 'w') as f:
            f.write('\n'.join(report))
    
    def run_full_experiment(self, num_workers: int = 4):
        """Execute the complete experiment workflow."""
        logger.info("Starting full experiment execution")
        
        # Run experiments
        results_df = self.run_experiment(num_workers)
        
        # Analyze results
        analysis = self.analyze_results(results_df)
        
        # Create visualizations
        self.create_visualizations(results_df)
        
        # Generate report
        self.generate_report(results_df, analysis)
        
        logger.info("Experiment execution completed")
        return results_df, analysis
    
if __name__ == "__main__":
    # Add process safety for macOS
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # Run the complete experiment
    experiment = Experiment1Runner()
    try:
        results, analysis = experiment.run_full_experiment(num_workers=4)
        print("Experiment completed successfully")
        print(f"Results saved to: {experiment.output_dir}")
    except Exception as e:
        print(f"Error running experiment: {str(e)}")