import carla
import gymnasium as gym
import numpy as np
import cv2
import random
import time
from gymnasium import spaces
import pygame

class CarlaEnv(gym.Env):
    def __init__(self, host='localhost', port=2000, town='Town01', 
                 im_width=640, im_height=480, show_preview=True):
        super(CarlaEnv, self).__init__()
        
        # CARLA connection
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set town
        if self.world.get_map().name != town:
            self.world = self.client.load_world(town)
        
        # Get blueprint library and spawn points
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        
        # Vehicle and sensors
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        
        # Environment parameters
        self.im_width = im_width
        self.im_height = im_height
        self.show_preview = show_preview
        
        # Episode tracking
        self.episode_start = time.time()
        self.collision_hist = []
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )  # [steering, throttle/brake]
        
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.im_height, self.im_width, 3),
            dtype=np.uint8
        )
        
        # Initialize pygame for preview
        if self.show_preview:
            pygame.init()
            self.display = pygame.display.set_mode((self.im_width, self.im_height))
            pygame.display.set_caption("CARLA PPO Training")
        
        self.front_camera = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Clean up existing actors
        self.cleanup()
        
        # Spawn vehicle
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Setup camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.im_width))
        camera_bp.set_attribute('image_size_y', str(self.im_height))
        camera_bp.set_attribute('fov', '110')
        
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(self.process_img)
        
        # Setup collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, 
                                                      carla.Transform(), 
                                                      attach_to=self.vehicle)
        self.collision_sensor.listen(self.collision_data)
        
        # Reset episode variables
        self.episode_start = time.time()
        self.collision_hist = []
        
        # Wait for sensors to initialize
        time.sleep(2)
        
        # Get initial observation
        while self.front_camera is None:
            time.sleep(0.01)
            
        return self.front_camera, {}
    
    def step(self, action):
        # Apply action
        steering = float(action[0])
        throttle_brake = float(action[1])
        
        if throttle_brake >= 0:
            throttle = throttle_brake
            brake = 0
        else:
            throttle = 0
            brake = -throttle_brake
            
        # Apply control to vehicle
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steering
        control.brake = brake
        self.vehicle.apply_control(control)
        
        # Get vehicle state
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h
        
        # Calculate reward
        reward = self.calculate_reward(speed)
        
        # Check if episode should terminate
        done = self.is_done()
        
        # Get observation
        obs = self.front_camera
        
        info = {
            'speed': speed,
            'collision': len(self.collision_hist) > 0,
            'episode_length': time.time() - self.episode_start
        }
        
        return obs, reward, done, False, info
    
    def calculate_reward(self, speed):
        reward = 0
        
        # Speed reward (encourage movement)
        if speed < 5:
            reward -= 1
        elif 5 <= speed <= 30:
            reward += speed / 30.0
        else:
            reward += 1 - ((speed - 30) / 50)  # Penalize excessive speed
        
        # Collision penalty
        if len(self.collision_hist) > 0:
            reward -= 100
            
        # Time-based reward (encourage longer episodes)
        reward += 0.1
        
        return reward
    
    def is_done(self):
        # Episode ends on collision
        if len(self.collision_hist) > 0:
            return True
            
        # Episode ends after maximum time
        if time.time() - self.episode_start > 300:  # 5 minutes
            return True
            
        return False
    
    def process_img(self, image):
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self.im_height, self.im_width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        self.front_camera = array
        
        # Show preview if enabled
        if self.show_preview:
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
            pygame.display.flip()
    
    def collision_data(self, event):
        self.collision_hist.append(event)
    
    def cleanup(self):
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
            
    def close(self):
        self.cleanup()
        if self.show_preview:
            pygame.quit()