# Experiment 1: Solution Quality Assessment Report

## Overview
- Total test instances: 810
- Capacity levels: [3, 5, 7]
- Demand scenarios: ['base', 'high', 'low']
- Market conditions: ['budget', 'luxury', 'standard']
- Replications per configuration: 30

## Overall Results
- Mean revenue gap: 3.09%
- Revenue gap 95% CI: [2.90%, 3.28%]
- Mean DP solution time: 1.16 seconds
- Mean SAA solution time: 13.91 seconds

## Statistical Analysis
- T-statistic: 25.3234
- P-value: 0.0000

## Results by Capacity Level

Capacity = 3
- Mean revenue gap: 3.40%
- Revenue gap std: 3.28%
- Mean DP time: 0.33 seconds
- Mean SAA time: 14.11 seconds

Capacity = 5
- Mean revenue gap: 3.01%
- Revenue gap std: 2.45%
- Mean DP time: 1.01 seconds
- Mean SAA time: 14.12 seconds

Capacity = 7
- Mean revenue gap: 2.87%
- Revenue gap std: 2.50%
- Mean DP time: 2.13 seconds
- Mean SAA time: 13.51 seconds

## Results by Demand Scenario

Scenario: base
- Mean revenue gap: 3.07%
- Revenue gap std: 2.84%
- Mean DP time: 1.16 seconds
- Mean SAA time: 13.99 seconds

Scenario: high
- Mean revenue gap: 3.07%
- Revenue gap std: 2.82%
- Mean DP time: 1.14 seconds
- Mean SAA time: 13.98 seconds

Scenario: low
- Mean revenue gap: 3.13%
- Revenue gap std: 2.66%
- Mean DP time: 1.17 seconds
- Mean SAA time: 13.77 seconds

## Results by Market Condition

Market: budget
- Mean revenue gap: 3.32%
- Revenue gap std: 2.18%
- Mean DP time: 0.90 seconds
- Mean SAA time: 13.85 seconds

Market: luxury
- Mean revenue gap: 5.39%
- Revenue gap std: 1.81%
- Mean DP time: 1.60 seconds
- Mean SAA time: 14.04 seconds

Market: standard
- Mean revenue gap: 0.56%
- Revenue gap std: 1.82%
- Mean DP time: 0.97 seconds
- Mean SAA time: 13.86 seconds