# Experiment 1: Solution Quality Assessment Report

## Overview
- Total test instances: 2700
- Capacity levels: [3, 5, 7]
- Demand scenarios: ['base', 'high', 'low']
- Market conditions: ['budget', 'luxury', 'standard']
- Replications per configuration: 100

## Overall Results
- Mean revenue gap: 3.72%
- Revenue gap 95% CI: [3.58%, 3.85%]
- Mean DP solution time: 370.26 seconds
- Mean SAA solution time: 36.17 seconds

## Statistical Analysis
- T-statistic: 41.1953
- P-value: 0.0000

## Results by Capacity Level

Capacity = 3
- Mean revenue gap: 5.48%
- Revenue gap std: 5.25%
- Mean DP time: 27.68 seconds
- Mean SAA time: 36.03 seconds

Capacity = 5
- Mean revenue gap: 2.95%
- Revenue gap std: 1.70%
- Mean DP time: 208.62 seconds
- Mean SAA time: 36.10 seconds

Capacity = 7
- Mean revenue gap: 2.71%
- Revenue gap std: 1.50%
- Mean DP time: 874.48 seconds
- Mean SAA time: 36.39 seconds

## Results by Demand Scenario

Scenario: base
- Mean revenue gap: 3.72%
- Revenue gap std: 3.53%
- Mean DP time: 371.63 seconds
- Mean SAA time: 36.23 seconds

Scenario: high
- Mean revenue gap: 3.72%
- Revenue gap std: 3.53%
- Mean DP time: 364.60 seconds
- Mean SAA time: 35.93 seconds

Scenario: low
- Mean revenue gap: 3.71%
- Revenue gap std: 3.53%
- Mean DP time: 374.57 seconds
- Mean SAA time: 36.35 seconds

## Results by Market Condition

Market: budget
- Mean revenue gap: 3.16%
- Revenue gap std: 1.31%
- Mean DP time: 260.78 seconds
- Mean SAA time: 36.19 seconds

Market: luxury
- Mean revenue gap: 6.72%
- Revenue gap std: 4.44%
- Mean DP time: 554.39 seconds
- Mean SAA time: 36.07 seconds

Market: standard
- Mean revenue gap: 1.26%
- Revenue gap std: 0.78%
- Mean DP time: 295.61 seconds
- Mean SAA time: 36.25 seconds