# Experiment 1: Solution Quality Assessment Report

## Overview
- Total test instances: 240
- Capacity levels: [3, 5]
- Demand scenarios: ['base', 'low']
- Market conditions: ['budget', 'standard']
- Replications per configuration: 30

## Overall Results
- Mean revenue gap: 1.66%
- Revenue gap 95% CI: [1.36%, 1.96%]
- Mean DP solution time: 0.51 seconds
- Mean SAA solution time: 13.61 seconds

## Statistical Analysis
- T-statistic: 9.8743
- P-value: 0.0000

## Results by Capacity Level

Capacity = 3
- Mean revenue gap: 1.63%
- Revenue gap std: 2.28%
- Mean DP time: 0.26 seconds
- Mean SAA time: 13.73 seconds

Capacity = 5
- Mean revenue gap: 1.69%
- Revenue gap std: 2.46%
- Mean DP time: 0.77 seconds
- Mean SAA time: 13.48 seconds

## Results by Demand Scenario

Scenario: base
- Mean revenue gap: 1.82%
- Revenue gap std: 2.24%
- Mean DP time: 0.51 seconds
- Mean SAA time: 13.84 seconds

Scenario: low
- Mean revenue gap: 1.51%
- Revenue gap std: 2.48%
- Mean DP time: 0.52 seconds
- Mean SAA time: 13.37 seconds

## Results by Market Condition

Market: budget
- Mean revenue gap: 2.88%
- Revenue gap std: 2.12%
- Mean DP time: 0.49 seconds
- Mean SAA time: 13.55 seconds

Market: standard
- Mean revenue gap: 0.45%
- Revenue gap std: 1.94%
- Mean DP time: 0.54 seconds
- Mean SAA time: 13.66 seconds