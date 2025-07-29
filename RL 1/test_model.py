import numpy as np
import time
from stable_baselines3 import PPO
from carla_env import CarlaEnv
import argparse

def test_trained_model(model_path, num_episodes=5, render=True, max_steps=1000):
    """Test the trained PPO model with debugging"""
    
    # Create environment with preview enabled
    env = CarlaEnv(
        host='localhost',
        port=2000,
        town='Town01',
        im_width=400,
        im_height=400,
        show_preview=render
    )
    
    # Load trained model
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    # Test for multiple episodes
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        stuck_counter = 0  # Track if vehicle is stuck
        
        while not done and episode_length < max_steps:
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Debug: Print action values
            if episode_length % 100 == 0:
                print(f"Action: Steering={action[0]:.3f}, Throttle/Brake={action[1]:.3f}")
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Check if vehicle is stuck (not moving)
            if info['speed'] < 1.0:
                stuck_counter += 1
            else:
                stuck_counter = 0
                
            # Force end episode if stuck for too long
            if stuck_counter > 200:  # 200 steps without movement
                print(f"Vehicle stuck for {stuck_counter} steps, ending episode")
                done = True
            
            # Print information
            if episode_length % 100 == 0:
                print(f"Step {episode_length}: Speed={info['speed']:.1f} km/h, "
                      f"Reward={reward:.2f}, Collision={info['collision']}")
                print(f"Stuck counter: {stuck_counter}")
            
            # Small delay for visualization
            if render:
                time.sleep(0.05)
        
        print(f"Episode {episode + 1} finished!")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Episode length: {episode_length} steps")
        print(f"Final speed: {info['speed']:.1f} km/h")
        print(f"Collision occurred: {info['collision']}")
        print(f"Episode ended due to: {'Max steps' if episode_length >= max_steps else 'Natural termination'}")
        
        # Wait between episodes
        time.sleep(2)
    
    env.close()
    print("\nTesting completed!")

def debug_model_actions(model_path, num_steps=100):
    """Debug what actions the model is predicting"""
    
    print("=== Debugging Model Actions ===")
    
    # Create environment
    env = CarlaEnv(show_preview=False)
    
    # Load model
    model = PPO.load(model_path)
    
    # Reset environment
    obs, _ = env.reset()
    
    actions_log = []
    
    for step in range(num_steps):
        # Get action
        action, _states = model.predict(obs, deterministic=True)
        actions_log.append(action.copy())
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        if step % 10 == 0:
            print(f"Step {step}: Steering={action[0]:.3f}, Throttle/Brake={action[1]:.3f}, "
                  f"Speed={info['speed']:.1f} km/h, Reward={reward:.2f}")
        
        if done:
            break
    
    env.close()
    
    # Analyze actions
    actions_array = np.array(actions_log)
    print(f"\n=== Action Analysis ===")
    print(f"Steering - Mean: {actions_array[:, 0].mean():.3f}, Std: {actions_array[:, 0].std():.3f}")
    print(f"Throttle/Brake - Mean: {actions_array[:, 1].mean():.3f}, Std: {actions_array[:, 1].std():.3f}")
    print(f"Actions near zero: {np.sum(np.abs(actions_array) < 0.1) / (len(actions_log) * 2) * 100:.1f}%")

def test_random_actions(num_steps=100):
    """Test with random actions to verify environment works"""
    
    print("=== Testing Random Actions ===")
    
    env = CarlaEnv(show_preview=True)
    obs, _ = env.reset()
    
    for step in range(num_steps):
        # Random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step}: Steering={action[0]:.3f}, Throttle/Brake={action[1]:.3f}, "
                  f"Speed={info['speed']:.1f} km/h, Reward={reward:.2f}")
        
        time.sleep(0.1)
        
        if done:
            break
    
    env.close()
    print("Random actions test completed!")

def evaluate_model_performance(model_path, num_episodes=10):
    """Evaluate model performance over multiple episodes"""
    
    env = CarlaEnv(show_preview=False)  # No preview for evaluation
    model = PPO.load(model_path)
    
    episode_rewards = []
    episode_lengths = []
    collision_count = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info['collision']:
            collision_count += 1
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Length={episode_length}, Collision={info['collision']}")
    
    env.close()
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    collision_rate = collision_count / num_episodes
    
    print(f"\n--- Performance Summary ---")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.1f}")
    print(f"Collision Rate: {collision_rate:.2%}")
    
    return {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'collision_rate': collision_rate,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained PPO model on CARLA')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                       help='Show visual output')
    parser.add_argument('--debug', action='store_true',
                       help='Run debug mode to analyze model actions')
    parser.add_argument('--test_random', action='store_true',
                       help='Test with random actions instead of model')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    if args.debug:
        debug_model_actions(args.model_path)
    elif args.test_random:
        test_random_actions()
    else:
        test_trained_model(args.model_path, args.episodes, args.render, args.max_steps)