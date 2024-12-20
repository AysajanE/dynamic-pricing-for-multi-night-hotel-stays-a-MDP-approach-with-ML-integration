# Experiment 1: DP vs. SAA Solution Quality Assessment
# Full Scale Implementation

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
import os

from data_generator import TestConfiguration, create_test_instance
from dynamic_pricing_algorithms import DynamicProgramming, StochasticApproximation, DPState

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
    
    Initialize experiment runner with explicit assumptions:
        1. Both algorithms use identical evaluation paths
        2. Revenue calculations match exactly
        3. Capacity constraints are handled identically
        4. Training randomization does not affect evaluation
    """
    
    def __init__(self, base_results_dir: str = "../experiments/experiment1"):
        """
        Initialize experiment runner with configuration.
        
        Args:
            base_results_dir: Path to the base results directory, relative to the script location
        """
        # Get the directory where the script is located
        script_dir = Path(__file__).parent.absolute()
        
        # Construct the full path to the results directory
        self.base_results_dir = Path(script_dir) / base_results_dir
        self.output_dir = self.base_results_dir / "results"
        
        # Create the experiment directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log the results directory location
        logger.info(f"Results will be saved to: {self.output_dir}")
        
        # Define experiment parameters
        self.T = 10  # Booking horizon
        self.N = 5  # Service horizon
        
        # Define parameter ranges for test instances
        self.capacity_levels = [3, 5, 7]  # Small capacities for tractable DP
        self.demand_scenarios = ['low', 'base', 'high']
        self.market_conditions = ['budget', 'standard', 'luxury']
        
        # self.capacity_levels = [3]  # Small capacities for tractable DP
        # self.demand_scenarios = ['base']
        # self.market_conditions = ['standard']
        
        # SAA learning parameters
        self.learning_params = {
            'eta_0': 0.3,        # Initial learning rate
            'gamma': 0.05,       # Learning rate decay
            'eta_min': 0.001,    # Minimum learning rate
            'max_epochs': 1000,  # Maximum training epochs
            'batch_size': 64     # Mini-batch size
        }
        
        # Statistical parameters
        self.num_replications = 100  # Number of replications per configuration
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
    
    def _validate_revenue_calculations(self,
                                    path: List[Tuple],
                                    dp_policy: Dict,
                                    dp_revenue: float,
                                    saa_prices: Dict,
                                    saa_revenue: float,
                                    instance: Dict,
                                    tolerance: float = 1e-10) -> bool:
        """
        Verify that revenue calculations match exactly between DP and SAA evaluations.

        Args:
            path: Evaluation path
            dp_policy: DP policy mapping states to prices
            dp_revenue: Total revenue from DP evaluation
            saa_prices: SAA price vectors
            saa_revenue: Total revenue from SAA evaluation
            instance: Test instance dictionary containing problem parameters
            tolerance: Numerical tolerance for floating point comparisons

        Returns:
            bool: True if revenue calculations are consistent
        """
        # Get parameters from instance
        N = instance['parameters'].N
        C = instance['parameters'].C

        # Recalculate revenues step by step
        dp_revenue_check = 0.0
        saa_revenue_check = 0.0
        capacity = np.full(N, C, dtype=np.int32)

        for t, bt, qt in path:
            if bt is not None:
                # Check DP revenue calculation
                state = DPState(capacity=tuple(capacity), time=t)
                if state in dp_policy:
                    stay_nights = instance['booking_classes']
                    if all(capacity[i] >= 1 for i in stay_nights):
                        dp_prices = [dp_policy[state][i] for i in stay_nights]
                        dp_avg_price = sum(dp_prices) / len(stay_nights)
                        if qt >= dp_avg_price:
                            dp_revenue_check += sum(dp_prices)

                # Check SAA revenue calculation
                stay_nights = instance['booking_classes']
                if all(capacity[i] >= 1 for i in stay_nights):
                    saa_prices_stay = [saa_prices[t][i] for i in stay_nights]
                    saa_avg_price = sum(saa_prices_stay) / len(stay_nights)
                    if qt >= saa_avg_price:
                        saa_revenue_check += sum(saa_prices_stay)

        # Verify calculations match reported revenues
        dp_match = abs(dp_revenue - dp_revenue_check) < tolerance
        saa_match = abs(saa_revenue - saa_revenue_check) < tolerance

        if not (dp_match and saa_match):
            logger.warning("Revenue calculation mismatch detected")
            logger.warning(f"DP Revenue: reported={dp_revenue:.2f}, calculated={dp_revenue_check:.2f}")
            logger.warning(f"SAA Revenue: reported={saa_revenue:.2f}, calculated={saa_revenue_check:.2f}")

        return dp_match and saa_match
    
    def _enhanced_path_validation(self, paths: List, instance: Dict):
        """
        Enhanced validation of evaluation paths to ensure complete consistency.

        Args:
            paths: List of evaluation paths
            instance: Test instance dictionary

        Raises:
            ValueError: If any validation check fails
        """
        if not paths:
            raise ValueError("Empty evaluation paths")

        for path in paths:
            # Verify path length
            if len(path) != instance['parameters'].T:
                raise ValueError(f"Path length {len(path)} does not match horizon {instance['parameters'].T}")

            # Verify each component
            for t, bt, qt in path:
                # Verify time period
                if not (1 <= t <= instance['parameters'].T):
                    raise ValueError(f"Invalid time period {t}")

                if bt is not None:
                    # Verify booking class
                    if bt not in instance['booking_classes']:
                        raise ValueError(f"Invalid booking class {bt}")

                    # Verify reservation price
                    epsilon = instance['reservation_price_params'][bt]
                    if not (0 <= qt <= 1/epsilon):
                        raise ValueError(f"Invalid reservation price {qt}")

        logger.info("Enhanced path validation completed successfully")
        
    def _verify_evaluation_consistency(self,
                                    evaluation_paths: List,
                                    dp_policy: Dict,
                                    dp_revenue: float,
                                    saa_prices: Dict,
                                    saa_revenue: float,
                                    instance: Dict) -> bool:
        """
        Comprehensive verification of evaluation consistency between DP and SAA.

        Args:
            evaluation_paths: List of paths used for evaluation
            dp_policy: DP policy mapping states to prices
            dp_revenue: Total revenue from DP evaluation
            saa_prices: SAA price vectors
            saa_revenue: Total revenue from SAA evaluation
            instance: Test instance dictionary

        Returns:
            bool: True if all consistency checks pass
        """
        try:
            # Verify revenue calculations
            revenue_consistent = self._validate_revenue_calculations(
                evaluation_paths[0],  # Check first path as sample
                dp_policy,
                dp_revenue,
                saa_prices,
                saa_revenue,
                instance
            )

            if not revenue_consistent:
                logger.error("Revenue calculation consistency check failed")
                return False

            return True

        except Exception as e:
            logger.error(f"Error in evaluation consistency check: {str(e)}")
            return False
        
    def run_single_instance(self, instance: Dict, replication: int, eval_seed: int = None) -> Dict:
        """Run both DP and SAA on a single test instance with shared sample paths."""
        try:
            # Initialize SAA
            saa = StochasticApproximation(instance, self.learning_params)

            # Set evaluation seed
            eval_rng = np.random.default_rng(eval_seed if eval_seed is not None 
                                           else 1000 * replication)

            # Generate evaluation paths
            num_eval_paths = 1000
            evaluation_paths = []
            for _ in range(num_eval_paths):
                path = saa._generate_sample_path(rng=eval_rng)
                evaluation_paths.append(path)

            # Enhanced validation
            self._enhanced_path_validation(evaluation_paths, instance)

            # Solve using DP
            dp = DynamicProgramming(instance)
            dp_start = datetime.now()
            dp_policy, _ = dp.solve()
            dp_time = (datetime.now() - dp_start).total_seconds()

            # Evaluate DP policy
            dp_revenue = dp.evaluate_policy(dp_policy, evaluation_paths)

            # Solve using SAA
            saa_start = datetime.now()
            saa_prices, _, saa_time = saa.solve()

            # Evaluate SAA policy
            saa_revenue = saa.evaluate(saa_prices, evaluation_paths)

            # Verify evaluation consistency
            if not self._verify_evaluation_consistency(
                evaluation_paths,
                dp_policy,
                dp_revenue,
                saa_prices,
                saa_revenue,
                instance  # Pass instance to verification method
            ):
                logger.warning("Evaluation consistency check failed")

            # Calculate revenue gap
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
                'revenue_gap': revenue_gap,
                'num_eval_paths': num_eval_paths,
                'eval_seed': eval_seed if eval_seed is not None else 1000 * replication
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
            'mean_revenue_gap': float(results_df['revenue_gap'].mean()),  # Convert to float
            'std_revenue_gap': float(results_df['revenue_gap'].std()),
            'mean_dp_time': float(results_df['dp_time'].mean()),
            'mean_saa_time': float(results_df['saa_time'].mean())
        }

        # Paired t-test for revenue differences
        t_stat, p_value = stats.ttest_rel(
            results_df['dp_revenue'],
            results_df['saa_revenue']
        )

        analysis['statistical_tests'] = {
            't_statistic': float(t_stat),  # Convert to float
            'p_value': float(p_value)
        }

        # Confidence intervals for revenue gap
        ci = stats.t.interval(
            self.confidence_level,
            len(results_df) - 1,
            loc=results_df['revenue_gap'].mean(),
            scale=stats.sem(results_df['revenue_gap'])
        )

        analysis['confidence_intervals'] = {
            'revenue_gap_lower': float(ci[0]),  # Convert to float
            'revenue_gap_upper': float(ci[1])
        }

        # Analysis by capacity level
        capacity_analysis = results_df.groupby('capacity').agg({
            'revenue_gap': ['mean', 'std'],
            'dp_time': 'mean',
            'saa_time': 'mean'
        })

        # Convert the nested dictionary structure to a more JSON-serializable format
        analysis['by_capacity'] = {
            str(cap): {  # Convert capacity to string
                'revenue_gap_mean': float(stats['revenue_gap']['mean']),
                'revenue_gap_std': float(stats['revenue_gap']['std']),
                'dp_time_mean': float(stats['dp_time']['mean']),
                'saa_time_mean': float(stats['saa_time']['mean'])
            }
            for cap, stats in capacity_analysis.iterrows()
        }

        # Save analysis results
        with open(self.output_dir / 'analysis_results.json', 'w') as f:
            json.dump(analysis, f, indent=4)

        return analysis

    def generate_report(self, results_df: pd.DataFrame, analysis: Dict):
        """Generate a comprehensive report of experimental results."""
        try:
            report = []
            report.append("# Experiment 1: Solution Quality Assessment Report")
            report.append("\n## Overview")
            report.append(f"- Total test instances: {len(results_df)}")
            report.append(f"- Capacity levels: {sorted(results_df['capacity'].unique())}")
            report.append(f"- Demand scenarios: {sorted(results_df['demand_scenario'].unique())}")
            report.append(f"- Market conditions: {sorted(results_df['market_condition'].unique())}")
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
            for capacity in sorted(results_df['capacity'].unique()):
                cap_stats = analysis['by_capacity'][str(capacity)]
                report.append(f"\nCapacity = {capacity}")
                report.append(f"- Mean revenue gap: {cap_stats['revenue_gap_mean']:.2f}%")
                report.append(f"- Revenue gap std: {cap_stats['revenue_gap_std']:.2f}%")
                report.append(f"- Mean DP time: {cap_stats['dp_time_mean']:.2f} seconds")
                report.append(f"- Mean SAA time: {cap_stats['saa_time_mean']:.2f} seconds")

            # Add summary statistics by demand scenario
            report.append("\n## Results by Demand Scenario")
            demand_stats = results_df.groupby('demand_scenario').agg({
                'revenue_gap': ['mean', 'std'],
                'dp_time': 'mean',
                'saa_time': 'mean'
            })

            for scenario in sorted(results_df['demand_scenario'].unique()):
                stats = demand_stats.loc[scenario]
                report.append(f"\nScenario: {scenario}")
                report.append(f"- Mean revenue gap: {stats['revenue_gap']['mean']:.2f}%")
                report.append(f"- Revenue gap std: {stats['revenue_gap']['std']:.2f}%")
                report.append(f"- Mean DP time: {stats['dp_time']['mean']:.2f} seconds")
                report.append(f"- Mean SAA time: {stats['saa_time']['mean']:.2f} seconds")

            # Add summary statistics by market condition
            report.append("\n## Results by Market Condition")
            market_stats = results_df.groupby('market_condition').agg({
                'revenue_gap': ['mean', 'std'],
                'dp_time': 'mean',
                'saa_time': 'mean'
            })

            for market in sorted(results_df['market_condition'].unique()):
                stats = market_stats.loc[market]
                report.append(f"\nMarket: {market}")
                report.append(f"- Mean revenue gap: {stats['revenue_gap']['mean']:.2f}%")
                report.append(f"- Revenue gap std: {stats['revenue_gap']['std']:.2f}%")
                report.append(f"- Mean DP time: {stats['dp_time']['mean']:.2f} seconds")
                report.append(f"- Mean SAA time: {stats['saa_time']['mean']:.2f} seconds")

            # Save report
            with open(self.output_dir / 'experiment_report.md', 'w') as f:
                f.write('\n'.join(report))

            logger.info("Report generated successfully")

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            logger.error("Results DataFrame head:")
            logger.error(results_df.head())
            logger.error("Analysis structure:")
            logger.error(json.dumps(analysis, indent=2))
            raise
            
    def create_visualizations(self, results_df: pd.DataFrame):
        """Create and save visualizations of experimental results."""
        import matplotlib
        matplotlib.use('Agg')  # Set backend to non-interactive

        # Reset any existing plots
        plt.close('all')

        try:
            # 1. Revenue Comparison Visualization
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            revenue_plot = sns.barplot(
                data=results_df,
                x='capacity',
                y='dp_revenue',
                hue='demand_scenario',
                errorbar=('ci', 95),
                ax=ax1
            )
            ax1.set_title('Dynamic Programming Revenue by Capacity and Demand Scenario')
            ax1.set_xlabel('Capacity Level')
            ax1.set_ylabel('Revenue')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'revenue_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig1)
            logger.info("Generated revenue comparison plot")

            # 2. Revenue Gap Distribution
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            gap_plot = sns.boxplot(
                data=results_df,
                x='capacity',
                y='revenue_gap',
                hue='market_condition',
                ax=ax2
            )
            ax2.set_title('Revenue Gap Distribution by Capacity and Market Condition')
            ax2.set_xlabel('Capacity Level')
            ax2.set_ylabel('Revenue Gap (%)')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'revenue_gap_distribution.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
            logger.info("Generated revenue gap distribution plot")

            # 3. Solution Time Comparison
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            time_data = pd.melt(
                results_df,
                id_vars=['capacity'],
                value_vars=['dp_time', 'saa_time'],
                var_name='Algorithm',
                value_name='Time (seconds)'
            )
            time_plot = sns.boxplot(
                data=time_data,
                x='capacity',
                y='Time (seconds)',
                hue='Algorithm',
                ax=ax3
            )
            ax3.set_title('Solution Time Comparison by Algorithm')
            ax3.set_xlabel('Capacity Level')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'solution_time_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig3)
            logger.info("Generated solution time comparison plot")

            logger.info("Successfully created all visualizations")

        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            # Print additional debugging information
            logger.error("Matplotlib version: " + matplotlib.__version__)
            logger.error("Seaborn version: " + sns.__version__)
            raise

        finally:
            # Ensure all plots are closed
            plt.close('all')
    
    def run_full_experiment(self, num_workers: int = 4):
        """Execute the complete experiment workflow."""
        logger.info("Starting full experiment execution")

        try:
            # Run experiments
            results_df = self.run_experiment(num_workers)
            logger.info("Experiments completed successfully")

            # Analyze results
            analysis = self.analyze_results(results_df)
            logger.info("Analysis completed successfully")

            # Generate report
            self.generate_report(results_df, analysis)
            logger.info("Report generation completed successfully")

            # Create visualizations
            self.create_visualizations(results_df)
            logger.info("Visualization creation completed successfully")

            logger.info("Experiment execution completed")
            return results_df, analysis

        except Exception as e:
            logger.error(f"Error in experiment execution: {str(e)}")
            raise
    
if __name__ == "__main__":
    # Add process safety for macOS
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # Run the complete experiment
    experiment = Experiment1Runner()
    try:
        results, analysis = experiment.run_full_experiment(num_workers=8)
        print("Experiment completed successfully")
        print(f"Results saved to: {experiment.output_dir}")
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
