# A Unified Methodology of Rare Event Prediction in Temporal Point Process

## Introduction
Rare events in marked temporal point processes (TPP) pose fundamental challenges due to severe imbalance, temporal dependence, and event rarity. We propose a universal rare event prediction framework that jointly models frequent background dynamics and residual event patterns, enabling generalization across a range of rarity thresholds. Our method decomposes sequences via a Hawkes-based residual filtering mechanism and integrates the resulting residuals with neural temporal point process models in a unified prediction architecture. We establish excess risk bounds under temporal data splitting, accounting for both residual estimation error and sample dependence. Extensive experiments on real-world and synthetic datasets show that the proposed approach consistently improves rare-event prediction performance across multiple state-of-the-art neural TPP baselines.

## Acknowledgments
This project uses code from [EasyTemporalPointProcess](https://github.com/ant-research/EasyTemporalPointProcess) and [ResidualTPP](https://github.com/ruoxinyuan/ResidualTPP).
