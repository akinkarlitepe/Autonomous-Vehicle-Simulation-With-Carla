import os
import sys
import numpy as np
import traceback
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from carla_env import CarlaEnv
import argparse

def test_carla_connection(host='localhost', port=2000):
    """Test CARLA connection before training"""
    print(f"Testing CARLA connection to {host}:{port}...")
    try:
        import carla
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()
        print("✓ CARLA connection successful!")
        print(f"✓ Current map: {world.get_map().name}")
        return True
    except Exception as e:
        print(f"✗ CARLA connection failed: {e}")
        print("Make sure CARLA server is running!")
        return False

def make_env(host='localhost', port=2000):
    """Create and return a CARLA environment"""
    def _init():
        print("Creating CARLA environment...")
        try:
            env = CarlaEnv(
                host=host,
                port=port,
                town='Town01',
                im_width=84,  # Smaller for faster training
                im_height=84,
                show_preview=False  # Disable preview during training
            )
            print("✓ Environment created successfully!")
            return Monitor(env)
        except Exception as e:
            print(f"✗ Failed to create environment: {e}")
            traceback.print_exc()
            raise e
    return _init

def train_ppo_agent(total_timesteps=100000, save_path="./models/", log_path="./logs/", 
                   host='localhost', port=2000):
    """Train PPO agent on CARLA environment"""
    
    print("=== CARLA PPO Training Started ===")
    
    # Test CARLA connection first
    if not test_carla_connection(host, port):
        print("Exiting due to CARLA connection failure.")
        return None
    
    # Create directories
    print("Creating directories...")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    print(f"✓ Models will be saved to: {save_path}")
    print(f"✓ Logs will be saved to: {log_path}")
    
    # Check device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    try:
        # Create environment
        print("Creating vectorized environment...")
        env = DummyVecEnv([make_env(host, port)])
        print("✓ Vectorized environment created!")
        
        # Test environment reset
        print("Testing environment reset...")
        obs = env.reset()
        print(f"✓ Environment reset successful! Observation shape: {obs.shape}")
        
        # Create PPO model
        print("Creating PPO model...")
        model = PPO(
            "CnnPolicy",  # CNN policy for image observations
            env,
            verbose=1,
            tensorboard_log=log_path,
            learning_rate=3e-4,
            n_steps=512,  # Reduced for faster initial training
            batch_size=32,  # Reduced for better stability
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            device=device
        )
        print("✓ PPO model created successfully!")
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,  # Save more frequently
            save_path=save_path,
            name_prefix='ppo_carla'
        )
        
        # Train the model
        print(f"Starting training for {total_timesteps} timesteps...")
        print("This may take a while. Monitor progress below:")
        print("-" * 50)
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(save_path, "ppo_carla_final")
        model.save(final_model_path)
        print(f"✓ Training completed! Model saved to {final_model_path}")
        
        return model
        
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        traceback.print_exc()
        return None

def quick_test_run(host='localhost', port=2000):
    """Run a quick test to verify everything works"""
    print("=== Running Quick Test ===")
    
    try:
        # Test environment creation
        env = CarlaEnv(host=host, port=port, show_preview=True)
        print("✓ Environment created")
        
        # Test reset
        obs, _ = env.reset()
        print(f"✓ Environment reset successful, obs shape: {obs.shape}")
        
        # Test a few random actions
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {i+1}: Reward={reward:.2f}, Speed={info['speed']:.1f} km/h")
            
            if done:
                break
        
        env.close()
        print("✓ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent on CARLA')
    parser.add_argument('--timesteps', type=int, default=50000, 
                       help='Total training timesteps')
    parser.add_argument('--save_path', type=str, default='./models/', 
                       help='Path to save models')
    parser.add_argument('--log_path', type=str, default='./logs/', 
                       help='Path for tensorboard logs')
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test only')
    
    args = parser.parse_args()
    
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if args.test:
        # Run quick test
        success = quick_test_run(args.host, args.port)
        if success:
            print("Test passed! You can now run full training.")
        else:
            print("Test failed! Please fix issues before training.")
    else:
        # Start training
        model = train_ppo_agent(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            log_path=args.log_path,
            host=args.host,
            port=args.port
        )
        
        if model is None:
            print("Training failed. Please check the errors above.")
            sys.exit(1)
        else:
            print("Training completed successfully!")