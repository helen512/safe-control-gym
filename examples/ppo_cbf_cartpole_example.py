"""
Complete Example: PPO with CBF Safety Filter on Cartpole Environment

This example demonstrates how to:
1. Set up a cartpole environment with safety constraints
2. Train a PPO controller
3. Apply a CBF safety filter to certify actions
4. Compare unsafe vs safe control
"""

from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from safe_control_gym.envs.benchmark_env import Environment, Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.registration import make


def train_ppo_controller(gui=False, save_model=True):
    """
    Step 1: Train a PPO controller on the cartpole environment
    
    Modules Used:
    - Environment: cartpole with physics simulation
    - PPO Controller: reinforcement learning agent
    """
    print("=" * 60)
    print("STEP 1: Training PPO Controller")
    print("=" * 60)
    
    # Configuration for cartpole environment
    task_config = {
        'seed': 42,
        'task': Task.STABILIZATION,
        'task_info': {
            'stabilization_goal': [0, 0],  # Goal: keep cart at position 0 with pole upright
            'stabilization_goal_tolerance': 0.05
        },
        'ctrl_freq': 25,
        'pyb_freq': 1000,
        'episode_len_sec': 5,  # Shorter episodes for testing
        'cost': 'rl_reward',
        
        # Define safety constraints
        'constraints': [
            {
                'constraint_form': 'default_constraint',
                'constrained_variable': 'state',
                'upper_bounds': [2, 2, 0.2, 2],  # [x, x_dot, theta, theta_dot]
                'lower_bounds': [-2, -2, -0.2, -2]
            },
            {
                'constraint_form': 'default_constraint',
                'constrained_variable': 'input'
            }
        ],
        
        # Initial state randomization
        'randomized_init': True,
        'init_state_randomization_info': {
            'init_x': {'distrib': 'uniform', 'low': -0.5, 'high': 0.5},
            'init_x_dot': {'distrib': 'uniform', 'low': -0.5, 'high': 0.5},
            'init_theta': {'distrib': 'uniform', 'low': -0.15, 'high': 0.15},
            'init_theta_dot': {'distrib': 'uniform', 'low': -0.5, 'high': 0.5}
        },
        'done_on_out_of_bound': True,
        'done_on_violation': False
    }
    
    # PPO algorithm configuration - complete set of required parameters
    ppo_config = {
        'training': True,
        # Model args
        'hidden_dim': 64,
        'activation': 'tanh',
        'norm_obs': False,
        'norm_reward': False,
        'clip_obs': 10,
        'clip_reward': 10,
        # Loss args
        'gamma': 0.99,
        'use_gae': True,
        'gae_lambda': 0.95,
        'use_clipped_value': False,
        'clip_param': 0.2,
        'target_kl': 0.01,
        'entropy_coef': 0.01,
        # Optim args
        'opt_epochs': 10,
        'mini_batch_size': 64,
        'actor_lr': 0.0003,
        'critic_lr': 0.001,
        'max_grad_norm': 0.5,
        # Runner args
        'max_env_steps': 5000,  # Training steps (reduced for testing)
        'num_workers': 1,
        'rollout_batch_size': 4,
        'rollout_steps': 100,
        'deque_size': 10,
        'eval_batch_size': 10,
        # Misc
        'log_interval': 1000,
        'save_interval': 0,
        'num_checkpoints': 0,
        'eval_interval': 0,
        'eval_save_best': False,
        'tensorboard': False
    }
    
    # Create environment factory
    env_func = partial(make, Environment.CARTPOLE, **task_config)
    
    # Create training and evaluation environments
    train_env = env_func(gui=False)
    eval_env = env_func(gui=gui)
    
    # Create PPO controller
    print("\nCreating PPO controller...")
    ppo_controller = make('ppo', env_func, **ppo_config, 
                         output_dir='./temp_ppo', 
                         checkpoint_path='./temp_ppo/model_checkpoint.pt')
    
    # Train the controller
    print("\nTraining PPO...")
    experiment = BaseExperiment(env=eval_env, ctrl=ppo_controller, train_env=train_env)
    experiment.launch_training()
    
    # Save trained model
    if save_model:
        ppo_controller.save('./models/ppo_cartpole_trained.pt')
        print("\nPPO model saved to './models/ppo_cartpole_trained.pt'")
    
    train_env.close()
    return ppo_controller, eval_env, env_func


def setup_cbf_filter(env_func, use_neural_network=False):
    """
    Step 2: Setup CBF Safety Filter
    
    Modules Used:
    - CBF: Analytical barrier function (model-based)
    - CBF_NN: Neural network enhanced barrier function (learning-based)
    
    The CBF ensures that actions satisfy: ḣ(x,u) ≥ -α(h(x))
    where h(x) is the barrier function defining the safe set.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Setting up CBF Safety Filter")
    print("=" * 60)
    
    if use_neural_network:
        print("\nUsing CBF_NN (Neural Network Enhanced)")
        # CBF_NN learns the error in the Lie derivative from data
        cbf_config = {
            'slope': 0.1,  # Slope of the class-K function α(h) = slope * h
            'soft_constrained': True,
            'slack_weight': 10000.0,
            'slack_tolerance': 1.0E-3,
            # Neural network parameters
            'max_num_steps': 250,
            'hidden_dims': [256, 256],
            'learning_rate': 0.001,
            'num_episodes': 20,
            'max_buffer_size': 1.0E+6,
            'train_batch_size': 64,
            'train_iterations': 200,
            'prior_info': {
                'prior_prop': None,
                'randomize_prior_prop': False,
                'prior_prop_rand_info': None
            }
        }
        safety_filter = make('cbf_nn', env_func, **cbf_config, output_dir='./temp_cbf')
    else:
        print("\nUsing CBF (Analytical)")
        # Standard CBF using known model dynamics
        cbf_config = {
            'slope': 0.1,
            'soft_constrained': True,
            'slack_weight': 10000.0,
            'slack_tolerance': 1.0E-3,
            'prior_info': {
                'prior_prop': None,
                'randomize_prior_prop': False,
                'prior_prop_rand_info': None
            }
        }
        safety_filter = make('cbf', env_func, **cbf_config, output_dir='./temp_cbf')
    
    print(f"\nCBF Configuration:")
    print(f"  - Slope (α): {cbf_config['slope']}")
    print(f"  - Soft Constrained: {cbf_config['soft_constrained']}")
    print(f"  - Slack Weight: {cbf_config['slack_weight']}")
    
    return safety_filter


def evaluate_with_and_without_safety(ppo_controller, safety_filter, env_func, n_episodes=5):
    """
    Step 3: Evaluate PPO with and without CBF
    
    This demonstrates how the BaseExperiment integrates:
    - Controller (PPO): Proposes actions
    - Safety Filter (CBF): Certifies/modifies actions for safety
    """
    print("\n" + "=" * 60)
    print("STEP 3: Comparing Safe vs Unsafe Control")
    print("=" * 60)
    
    # Evaluate without safety filter
    print("\n--- Evaluating PPO WITHOUT safety filter ---")
    env_unsafe = env_func(gui=False)
    experiment_unsafe = BaseExperiment(env=env_unsafe, ctrl=ppo_controller)
    results_unsafe, metrics_unsafe = experiment_unsafe.run_evaluation(n_episodes=n_episodes)
    env_unsafe.close()
    
    # Evaluate with safety filter
    print("\n--- Evaluating PPO WITH CBF safety filter ---")
    env_safe = env_func(gui=False)
    experiment_safe = BaseExperiment(env=env_safe, ctrl=ppo_controller, safety_filter=safety_filter)
    results_safe, metrics_safe = experiment_safe.run_evaluation(n_episodes=n_episodes)
    
    # Extract safety filter data
    cbf_data = results_safe['safety_filter_data']
    
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"\nWithout CBF:")
    print(f"  Average Return: {metrics_unsafe['average_return']:.3f}")
    print(f"  Average Episode Length: {metrics_unsafe['average_length']:.1f}")
    print(f"  Constraint Violations: {metrics_unsafe['average_constraint_violation']:.1f}")
    print(f"  Failure Rate: {metrics_unsafe['failure_rate']:.2%}")
    
    print(f"\nWith CBF:")
    print(f"  Average Return: {metrics_safe['average_return']:.3f}")
    print(f"  Average Episode Length: {metrics_safe['average_length']:.1f}")
    print(f"  Constraint Violations: {metrics_safe['average_constraint_violation']:.1f}")
    print(f"  Failure Rate: {metrics_safe['failure_rate']:.2%}")
    
    # Calculate action corrections
    corrections = [np.linalg.norm(cbf_data['correction'][i]) for i in range(len(cbf_data['correction']))]
    avg_correction = np.mean([np.mean(c) for c in corrections])
    print(f"\n  Average Action Correction: {avg_correction:.4f}")
    print(f"  Times Action Modified: {sum([np.sum(c > 1e-6) for c in corrections])}")
    
    env_safe.close()
    return results_unsafe, results_safe, metrics_unsafe, metrics_safe


def visualize_results(results_unsafe, results_safe):
    """
    Step 4: Visualize the comparison
    """
    print("\n" + "=" * 60)
    print("STEP 4: Generating Visualizations")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get first episode data
    obs_unsafe = results_unsafe['obs'][0]
    obs_safe = results_safe['obs'][0]
    action_unsafe = results_unsafe['action'][0]
    action_safe = results_safe['action'][0]
    cbf_data = results_safe['safety_filter_data']
    
    # State indices: [x, x_dot, theta, theta_dot]
    
    # Plot 1: Cart Position over time
    axes[0, 0].plot(obs_unsafe[:, 0], 'r--', label='Without CBF', alpha=0.7)
    axes[0, 0].plot(obs_safe[:, 0], 'b-', label='With CBF', linewidth=2)
    axes[0, 0].axhline(y=2.0, color='k', linestyle=':', label='Constraint')
    axes[0, 0].axhline(y=-2.0, color='k', linestyle=':')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Cart Position (x)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Cart Position Trajectory')
    
    # Plot 2: Pole Angle over time
    axes[0, 1].plot(obs_unsafe[:, 2], 'r--', label='Without CBF', alpha=0.7)
    axes[0, 1].plot(obs_safe[:, 2], 'b-', label='With CBF', linewidth=2)
    axes[0, 1].axhline(y=0.2, color='k', linestyle=':', label='Constraint')
    axes[0, 1].axhline(y=-0.2, color='k', linestyle=':')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Pole Angle (θ)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Pole Angle Trajectory')
    
    # Plot 3: Phase Portrait (x vs theta)
    axes[1, 0].plot(obs_unsafe[:, 0], obs_unsafe[:, 2], 'r--', label='Without CBF', alpha=0.7)
    axes[1, 0].plot(obs_safe[:, 0], obs_safe[:, 2], 'b-', label='With CBF', linewidth=2)
    axes[1, 0].scatter(obs_safe[0, 0], obs_safe[0, 2], color='g', s=100, marker='o', label='Start', zorder=5)
    # Draw constraint box
    axes[1, 0].plot([-2, 2, 2, -2, -2], [-0.2, -0.2, 0.2, 0.2, -0.2], 'k:', linewidth=2, label='Safe Set')
    axes[1, 0].set_xlabel('Cart Position (x)')
    axes[1, 0].set_ylabel('Pole Angle (θ)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('State Space Trajectory')
    
    # Plot 4: Actions comparison
    axes[1, 1].plot(action_unsafe, 'r--', label='Proposed Action', alpha=0.7)
    axes[1, 1].plot(action_safe, 'b-', label='Certified Action', linewidth=2)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Control Input')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Control Actions')
    
    plt.tight_layout()
    plt.savefig('ppo_cbf_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to 'ppo_cbf_comparison.png'")
    plt.show()


def explain_module_interaction():
    """
    Explain how different modules work together
    """
    explanation = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              HOW MODULES WORK TOGETHER: PPO + CBF                         ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    1. ENVIRONMENT (safe_control_gym.envs.benchmark_env)
       ├─ Provides: State observations, dynamics, constraints
       ├─ cartpole.py: Physics simulation using PyBullet
       └─ Defines: Safe set h(x) = {x | constraints satisfied}
    
    2. PPO CONTROLLER (safe_control_gym.controllers.ppo)
       ├─ Input: Current observation (state)
       ├─ Output: Proposed action u_des
       ├─ Training: Maximizes cumulative reward
       └─ Problem: May propose UNSAFE actions!
    
    3. CBF SAFETY FILTER (safe_control_gym.safety_filters.cbf)
       ├─ Input: State x, Proposed action u_des
       ├─ Output: Certified action u_safe
       ├─ Method: Solves Quadratic Program (QP)
       │
       │   minimize: ½||u - u_des||² + w·s²
       │   subject to: ḣ(x,u) ≥ -α(h(x)) - s
       │               u_min ≤ u ≤ u_max
       │               s ≥ 0  (slack variable)
       │
       └─ Guarantee: u_safe keeps system in safe set!
    
    4. INTEGRATION (BaseExperiment._select_action)
       
       Flow at each timestep:
       
       observation ─→ PPO Controller ─→ uncertified_action
                                              │
                                              ├─────→ CBF Filter
                                              │         │
                                              │      (Solves QP)
                                              │         │
                                              │    certified_action
                                              │         │
       Environment ←─────────────────────────┴─────────┘
    
    5. KEY FILES AND THEIR ROLES:
    
       base_safety_filter.py (line 12-28)
       └─ Abstract interface: certify_action(state, action) → safe_action
    
       cbf.py (line 16-337)
       ├─ Line 68-72: Creates barrier function for cartpole (ellipsoid)
       ├─ Line 85-94: Computes Lie derivative ḣ(x,u)
       ├─ Line 105-162: Sets up QP optimization problem
       └─ Line 217-242: certify_action() - main interface
    
       cbf_nn.py (line 19-386)
       ├─ Extends CBF with neural network
       ├─ Line 72-73: MLP learns Lie derivative error
       ├─ Line 310-386: learn() - trains NN from data
       └─ Useful when model has errors/uncertainties
    
       cbf_utils.py (line 1-218)
       ├─ Line 9-29: cbf_cartpole() - defines barrier function
       ├─ Line 32-43: linear_function() - class-K function
       └─ Line 64-218: CBFBuffer - replay buffer for CBF_NN
    
       base_experiment.py (line 167-194)
       └─ Line 177-184: Integration logic
           PPO proposes → CBF certifies → Environment executes
    
    6. CONFIGURATION FILES:
    
       cartpole_config.yaml
       └─ Environment: constraints, dynamics, initial states
    
       ppo_config.yaml
       └─ PPO: learning rate, network architecture, training steps
    
       cbf_config.yaml
       └─ CBF: slope α, slack weight, soft/hard constraints
    
    7. MATHEMATICAL GUARANTEE:
    
       If h(x₀) ≥ 0 (start in safe set) and we enforce:
           ḣ(x,u) ≥ -α(h(x)) for all time
       
       Then: h(x_t) ≥ 0 for all t → System stays SAFE!
    
       This is called "forward invariance" of the safe set.
    
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  KEY INSIGHT: CBF provides a SAFETY LAYER on top of any controller!      ║
    ║  Works with PPO, SAC, DDPG, or even simple PID controllers!              ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


def main(test_mode=False):
    """
    Main execution: Complete PPO + CBF pipeline
    """
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                 PPO + CBF Safety Filter Example                          ║")
    print("║                    Cartpole Environment                                  ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    
    if not test_mode:
        # Show module interaction explanation
        explain_module_interaction()
        input("\nPress Enter to start training PPO controller...")
    else:
        print("\n🧪 Running in TEST MODE - using reduced parameters for quick validation")
    
    # Step 1: Train PPO
    ppo_controller, env, env_func = train_ppo_controller(gui=False, save_model=True)
    
    # Step 2: Setup CBF
    safety_filter = setup_cbf_filter(env_func, use_neural_network=False)
    
    # Step 3: Evaluate (use fewer episodes in test mode)
    n_episodes = 2 if test_mode else 5
    results_unsafe, results_safe, metrics_unsafe, metrics_safe = \
        evaluate_with_and_without_safety(ppo_controller, safety_filter, env_func, n_episodes=n_episodes)
    
    # Step 4: Visualize (skip plots in test mode)
    if not test_mode:
        visualize_results(results_unsafe, results_safe)
    else:
        print("\n📊 Skipping visualization in test mode")
    
    # Cleanup
    ppo_controller.close()
    safety_filter.close()
    env.close()
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. PPO learns to control, but may violate constraints")
    print("2. CBF modifies actions to guarantee safety")
    print("3. Minimal performance loss, maximum safety gain")
    print("4. Modular: Can swap PPO with SAC, DDPG, etc.")
    print("=" * 60)


if __name__ == '__main__':
    import sys
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("🧪 Running quick test...")
        main(test_mode=True)
    else:
        main(test_mode=False)

