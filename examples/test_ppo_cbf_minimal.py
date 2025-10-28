#!/usr/bin/env python3
"""
Minimal test to verify PPO + CBF modules work correctly
"""

import sys
import os
from functools import partial

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from safe_control_gym.envs.benchmark_env import Environment, Task
        print("  ✓ Environment imports OK")
        
        from safe_control_gym.utils.registration import make
        print("  ✓ Registration utils OK")
        
        from safe_control_gym.experiments.base_experiment import BaseExperiment
        print("  ✓ Experiment framework OK")
        
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for testing
        import matplotlib.pyplot as plt
        print("  ✓ Required packages OK")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_environment_creation():
    """Test cartpole environment creation"""
    print("\n🏗️  Testing environment creation...")
    
    try:
        from safe_control_gym.envs.benchmark_env import Environment, Task
        from safe_control_gym.utils.registration import make
        
        # Minimal cartpole config
        task_config = {
            'seed': 42,
            'task': Task.STABILIZATION,
            'episode_len_sec': 2,  # Very short for testing
            'ctrl_freq': 10,
            'pyb_freq': 100,
            'constraints': [
                {
                    'constraint_form': 'default_constraint',
                    'constrained_variable': 'state',
                    'upper_bounds': [2, 2, 0.2, 2],
                    'lower_bounds': [-2, -2, -0.2, -2]
                }
            ]
        }
        
        # Create environment
        env_func = partial(make, Environment.CARTPOLE, **task_config)
        env = env_func(gui=False)
        
        # Test basic operations
        obs, info = env.reset()
        print(f"  ✓ Environment created - Initial obs shape: {obs.shape}")
        
        # Take a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"  ✓ Step {i+1}: obs shape={obs.shape}, reward={reward:.3f}")
            
        env.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Environment error: {e}")
        return False

def test_controller_creation():
    """Test PPO controller creation"""
    print("\n🧠 Testing PPO controller creation...")
    
    try:
        from safe_control_gym.envs.benchmark_env import Environment, Task
        from safe_control_gym.utils.registration import make
        
        # Minimal environment
        env_func = partial(make, Environment.CARTPOLE, 
                          seed=42, task=Task.STABILIZATION, episode_len_sec=1)
        
        # Complete PPO config based on default YAML
        ppo_config = {
            'training': False,  # Skip training for now
            # Model args
            'hidden_dim': 32,
            'activation': 'tanh',
            'norm_obs': False,
            'norm_reward': False,
            'clip_obs': 10,
            'clip_reward': 10,
            # Loss args
            'gamma': 0.99,
            'use_gae': False,
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
            'max_env_steps': 100,
            'num_workers': 1,
            'rollout_batch_size': 1,
            'rollout_steps': 10,
            'deque_size': 10,
            'eval_batch_size': 10,
            # Misc
            'log_interval': 0,
            'save_interval': 0,
            'num_checkpoints': 0,
            'eval_interval': 0,
            'eval_save_best': False,
            'tensorboard': False
        }
        
        # Create controller
        ctrl = make('ppo', env_func, **ppo_config, output_dir='./temp_test')
        print("  ✓ PPO controller created")
        
        # Test action selection
        env = env_func(gui=False)
        obs, info = env.reset()
        action = ctrl.select_action(obs, info)
        print(f"  ✓ Action selection works - action shape: {action.shape}")
        
        ctrl.close()
        env.close()
        
        # Cleanup temp directory
        import shutil
        shutil.rmtree('./temp_test', ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Controller error: {e}")
        return False

def test_cbf_creation():
    """Test CBF safety filter creation"""
    print("\n🛡️  Testing CBF safety filter creation...")
    
    try:
        from safe_control_gym.envs.benchmark_env import Environment, Task
        from safe_control_gym.utils.registration import make
        
        # Environment with constraints
        task_config = {
            'seed': 42,
            'task': Task.STABILIZATION,
            'episode_len_sec': 1,
            'constraints': [
                {
                    'constraint_form': 'default_constraint',
                    'constrained_variable': 'state',
                    'upper_bounds': [2, 2, 0.2, 2],
                    'lower_bounds': [-2, -2, -0.2, -2]
                },
                {
                    'constraint_form': 'default_constraint',
                    'constrained_variable': 'input'
                }
            ]
        }
        
        env_func = partial(make, Environment.CARTPOLE, **task_config)
        
        # CBF config
        cbf_config = {
            'slope': 0.1,
            'soft_constrained': True,
            'slack_weight': 10000.0,
            'prior_info': {'prior_prop': None}
        }
        
        # Create CBF
        cbf = make('cbf', env_func, **cbf_config, output_dir='./temp_cbf')
        print("  ✓ CBF safety filter created")
        
        # Test action certification
        env = env_func(gui=False)
        obs, info = env.reset()
        
        # Test with a sample action
        unsafe_action = env.action_space.sample()
        safe_action, success = cbf.certify_action(obs[:4], unsafe_action)  # Use only state part
        print(f"  ✓ Action certification works - success: {success}")
        
        cbf.close()
        env.close()
        
        # Cleanup
        import shutil
        shutil.rmtree('./temp_cbf', ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"  ❌ CBF error: {e}")
        return False

def test_integration():
    """Test integration of PPO + CBF through BaseExperiment"""
    print("\n🔗 Testing PPO + CBF integration...")
    
    try:
        from safe_control_gym.envs.benchmark_env import Environment, Task
        from safe_control_gym.utils.registration import make
        from safe_control_gym.experiments.base_experiment import BaseExperiment
        
        # Setup components
        task_config = {
            'seed': 42,
            'task': Task.STABILIZATION,
            'episode_len_sec': 1,
            'ctrl_freq': 10,
            'constraints': [
                {
                    'constraint_form': 'default_constraint',
                    'constrained_variable': 'state',
                    'upper_bounds': [2, 2, 0.2, 2],
                    'lower_bounds': [-2, -2, -0.2, -2]
                },
                {
                    'constraint_form': 'default_constraint',
                    'constrained_variable': 'input'
                }
            ]
        }
        
        env_func = partial(make, Environment.CARTPOLE, **task_config)
        env = env_func(gui=False)
        
        # Complete PPO config
        ppo_config = {
            'training': False,
            'hidden_dim': 16,
            'activation': 'tanh',
            'norm_obs': False,
            'norm_reward': False,
            'clip_obs': 10,
            'clip_reward': 10,
            'gamma': 0.99,
            'use_gae': False,
            'gae_lambda': 0.95,
            'use_clipped_value': False,
            'clip_param': 0.2,
            'target_kl': 0.01,
            'entropy_coef': 0.01,
            'opt_epochs': 10,
            'mini_batch_size': 64,
            'actor_lr': 0.0003,
            'critic_lr': 0.001,
            'max_grad_norm': 0.5,
            'max_env_steps': 50,
            'num_workers': 1,
            'rollout_batch_size': 1,
            'rollout_steps': 5,
            'deque_size': 10,
            'eval_batch_size': 10,
            'log_interval': 0,
            'save_interval': 0,
            'num_checkpoints': 0,
            'eval_interval': 0,
            'eval_save_best': False,
            'tensorboard': False
        }
        ctrl = make('ppo', env_func, **ppo_config, output_dir='./temp_integration')
        
        # Minimal CBF
        cbf_config = {
            'slope': 0.1,
            'soft_constrained': True,
            'prior_info': {'prior_prop': None}
        }
        cbf = make('cbf', env_func, **cbf_config, output_dir='./temp_cbf_int')
        
        # Test integration
        experiment = BaseExperiment(env=env, ctrl=ctrl, safety_filter=cbf)
        results, metrics = experiment.run_evaluation(n_episodes=1, verbose=False)
        
        print(f"  ✓ Integration test complete - collected {len(results['obs'])} episodes")
        print(f"  ✓ Safety filter data available: {'safety_filter_data' in results}")
        
        # Cleanup
        ctrl.close()
        cbf.close()
        env.close()
        experiment.close()
        
        import shutil
        shutil.rmtree('./temp_integration', ignore_errors=True)
        shutil.rmtree('./temp_cbf_int', ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 MINIMAL PPO + CBF FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Environment Creation", test_environment_creation),
        ("PPO Controller", test_controller_creation),
        ("CBF Safety Filter", test_cbf_creation),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The full example should work.")
        return True
    else:
        print("⚠️  Some tests failed. Check your installation.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
