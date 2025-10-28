# Quick Start Guide: PPO with CBF Safety Filter

This guide shows you how to use **Proximal Policy Optimization (PPO)** with **Control Barrier Function (CBF)** safety filters on the cartpole environment.

## 🚀 Quick Start

### Option 1: Run the Complete Example
```bash
cd /home/dmy/gymtest/safe-control-gym/examples/
python ppo_cbf_cartpole_example.py
```

### Option 2: Use Existing Framework Scripts

#### Step 1: Train PPO Controller
```bash
cd examples/rl/
python rl_experiment.py \
    --algo ppo \
    --task cartpole \
    --overrides \
        ./config_overrides/cartpole/cartpole_stab.yaml \
        ./config_overrides/cartpole/ppo_cartpole.yaml \
    --kv_overrides \
        algo_config.training=True \
        algo_config.max_env_steps=10000
```

#### Step 2: Test CBF Safety Filter
```bash
cd examples/cbf/
python cbf_experiment.py \
    --algo ppo \
    --task cartpole \
    --safety_filter cbf \
    --overrides \
        ./config_overrides/cartpole_config.yaml \
        ./config_overrides/ppo_config.yaml \
        ./config_overrides/cbf_config.yaml \
    --kv_overrides \
        training=False \
        n_episodes=5
```

## 📋 Prerequisites

```bash
# Install safe-control-gym
cd /home/dmy/gymtest/safe-control-gym/
pip install -e .

# Required packages
pip install torch matplotlib numpy casadi gymnasium pybullet
```

## 🔧 Module Overview

### Key Components

| Module | Purpose | File Location |
|--------|---------|---------------|
| **Environment** | Cartpole physics & constraints | `safe_control_gym/envs/gym_control/cartpole.py` |
| **PPO Controller** | RL policy learning | `safe_control_gym/controllers/ppo/ppo.py` |
| **CBF Filter** | Safety certification | `safe_control_gym/safety_filters/cbf/cbf.py` |
| **CBF_NN Filter** | Learning-enhanced safety | `safe_control_gym/safety_filters/cbf/cbf_nn.py` |
| **Experiment** | Integration framework | `safe_control_gym/experiments/base_experiment.py` |

### Configuration Files

```
examples/
├── rl/config_overrides/cartpole/
│   ├── cartpole_stab.yaml      # Environment settings
│   └── ppo_cartpole.yaml       # PPO hyperparameters
└── cbf/config_overrides/
    ├── cartpole_config.yaml    # Environment settings
    ├── ppo_config.yaml         # PPO settings for CBF
    ├── cbf_config.yaml         # Standard CBF settings
    └── cbf_nn_config.yaml      # Neural CBF settings
```

## 🎯 Understanding the Workflow

### 1. Environment Setup
- **Cartpole**: Inverted pendulum on cart
- **State**: `[x, x_dot, theta, theta_dot]` (position, velocity, angle, angular velocity)
- **Action**: Force applied to cart
- **Constraints**: Keep within safe bounds

### 2. PPO Controller
- **Goal**: Learn to balance pole upright
- **Training**: Maximizes reward through trial and error
- **Problem**: May propose unsafe actions that violate constraints

### 3. CBF Safety Filter
- **Input**: State + Proposed action
- **Process**: Solves optimization problem
  ```
  minimize: ||u_safe - u_proposed||²
  subject to: ḣ(x,u) ≥ -α(h(x))  [CBF constraint]
  ```
- **Output**: Certified safe action
- **Guarantee**: System stays within safe set

### 4. Integration via BaseExperiment
```python
# At each timestep:
obs = env.get_observation()
action_proposed = ppo_controller.select_action(obs)  
action_safe = cbf_filter.certify_action(obs, action_proposed)
obs_next = env.step(action_safe)
```

## 📊 What You'll See

### Training Output
```
STEP 1: Training PPO Controller
========================================
Creating PPO controller...
Training PPO...
Episode 100/500: Reward = 45.2, Length = 127
...
PPO model saved to './models/ppo_cartpole_trained.pt'

STEP 2: Setting up CBF Safety Filter
========================================
Using CBF (Analytical)
CBF Configuration:
  - Slope (α): 0.1
  - Soft Constrained: True
  - Slack Weight: 10000.0
```

### Evaluation Results
```
RESULTS COMPARISON
========================================
Without CBF:
  Average Return: 187.450
  Average Episode Length: 195.2
  Constraint Violations: 12.4
  Failure Rate: 80.00%

With CBF:
  Average Return: 181.230
  Average Episode Length: 189.8
  Constraint Violations: 0.0
  Failure Rate: 0.00%

  Average Action Correction: 0.0234
  Times Action Modified: 47
```

### Visualization
The script generates `ppo_cbf_comparison.png` showing:
1. **Cart Position**: How CBF keeps cart within bounds
2. **Pole Angle**: How CBF prevents pole from falling too far
3. **State Space**: 2D trajectory showing safe set boundary
4. **Actions**: Comparison of proposed vs certified actions

## 🛠 Customization Options

### Change CBF Parameters
```python
cbf_config = {
    'slope': 0.2,              # Higher = more conservative
    'soft_constrained': True,   # Allow small violations with penalty
    'slack_weight': 50000.0,   # Higher = stricter constraint enforcement
}
```

### Use Neural Network CBF
```python
# In setup_cbf_filter()
safety_filter = setup_cbf_filter(env_func, use_neural_network=True)
```

### Modify Environment Constraints
```python
'constraints': [
    {
        'constraint_form': 'default_constraint',
        'constrained_variable': 'state',
        'upper_bounds': [1.5, 2, 0.15, 2],  # Tighter position/angle limits
        'lower_bounds': [-1.5, -2, -0.15, -2]
    }
]
```

### Try Different Controllers
Replace `'ppo'` with:
- `'sac'` - Soft Actor-Critic
- `'ddpg'` - Deep Deterministic Policy Gradient
- `'pid'` - PID controller

## 🔍 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Install in development mode
pip install -e .
```

**2. CasADi Installation**
```bash
# If CasADi fails to install
conda install -c conda-forge casadi
```

**3. Training Takes Too Long**
```python
# Reduce training steps in ppo_config
'max_env_steps': 5000  # Instead of 50000
```

**4. No GUI Display**
```python
# Set gui=True to see visualization
ppo_controller, env, env_func = train_ppo_controller(gui=True)
```

### Performance Tips

1. **Faster Training**: Use smaller networks and fewer episodes
2. **Better Safety**: Increase `slack_weight` in CBF config
3. **Less Conservative**: Decrease `slope` parameter
4. **Debug Mode**: Set `verbose=True` in BaseExperiment

## 📚 Further Reading

- **CBF Theory**: [Control Barrier Functions: Theory and Applications](https://arxiv.org/abs/1903.11199)
- **CBF-NN**: [Learning for Safety-Critical Control with Control Barrier Functions](https://arxiv.org/abs/1912.10099)
- **Safe-Control-Gym**: [Repository Documentation](https://github.com/utiasDSL/safe-control-gym)

## 🎉 Next Steps

1. **Try Different Environments**: Quadrotor, other systems
2. **Experiment with CBF_NN**: Learn from real data
3. **Compare Multiple Controllers**: PPO vs SAC vs DDPG with CBF
4. **Custom Barrier Functions**: Design your own safety constraints

---

**Questions?** Check the example code in `ppo_cbf_cartpole_example.py` for detailed implementation!
