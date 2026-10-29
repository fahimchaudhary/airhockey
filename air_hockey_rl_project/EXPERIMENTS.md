# Experimental protocol

## Research questions
1. Does PPO outperform a rule-based baseline under identical game dynamics?
2. Does opponent curriculum improve generalization to harder opponents?
3. How robust is the learned policy to randomized puck speed, restitution, and friction?

## Baselines
- Random legal actions (sanity check only)
- Rule-based intercept controller (easy / medium / hard)
- PPO with fixed opponent
- PPO with curriculum (recommended extension)
- SAC under the same observation/action spaces and comparable environment steps

## Metrics
- Win rate and Wilson 95% confidence interval
- Mean goal difference
- Mean episodic return and standard deviation
- Goals per 1000 environment steps
- Training wall-clock time and environment steps to reach a target win rate
- Robustness under parameter perturbation

## Fair comparison
Use at least 5 independent seeds where feasible. Evaluate each checkpoint on a fixed set of held-out seeds and opponents. Do not tune hyperparameters on the held-out evaluation set. Report environment version, library versions, seeds, training steps, and all reward coefficients.

## Limitations
This simplified environment is for a student-scale demonstration. It is not a reproduction of the Robot Air Hockey Challenge and does not establish sim-to-real transfer or physical robot safety.
