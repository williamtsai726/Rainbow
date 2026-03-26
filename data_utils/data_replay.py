import ast
import concurrent
import json
import pickle
import time
from turtle import right
import torch
from tqdm import tqdm
import numpy as np
import os
from typing import Dict, Any, Optional
from gello.env import RobotEnv

# Import centralized logging utilities
from gello.utils.logging_utils import log_data_utils, log_replay, log_demo_data_info
import glob
from PIL import Image

# Add matplotlib imports for visualization
import matplotlib.pyplot as plt
from collections import deque

from gello.utils.logging_utils import log_collect_demos

class DataReplayer():
    """
    DataReplayer class for replaying episodes.

    Args:
        save_format: Format of the saved data (json or npy)
        old_format: Whether the data is in the old format (True) or new format (False)
    """
    def __init__(self, save_format, old_format):
        self.save_format = save_format
        self.old_format = old_format
        self.demo = None

        self.left_camera_key = "left_rgb"
        self.right_camera_key = "right_rgb"
        self.front_camera_key = "front_rgb"

        # for json format only
        # self.main_rgb_paths = None
        # self.wrist_rgb_paths = None
        self.left_rgb_paths = None
        self.right_rgb_paths = None
        self.front_rgb_paths = None
        
        # Visualization setup
        self.fig = None
        self.axs = None
        self.joint_line = None
        self.action_line = None
        self.history_len = 300
        self.joint_history = deque(maxlen=self.history_len)  # Keep last 100 joint positions
        self.step_history = deque(maxlen=self.history_len)    # Keep last 100 step indices
        
        # Initialize matplotlib backend for real-time plotting
        plt.ion()  # Turn on interactive mode
    
    def load_episode(self, root_dir, episode_number):
        # convention: 6 digits for episode number
        episode_number = f"{int(episode_number):06d}"
        try:
            # Check if file exists
            if self.save_format == "json":
                if not self.old_format:
                    demo_dir = f"{root_dir}/{episode_number}/{episode_number}.json"
                else:
                    demo_dir = f"{root_dir}_pickle/{episode_number}.pkl"
                if not os.path.exists(demo_dir):
                    log_data_utils(f"Episode file not found: {demo_dir}", "error")
                log_replay(f"Found episode file: {demo_dir}", "info")
                
                # Load demo data
                log_data_utils(f"Loading demo data from: {demo_dir}", "info")
                try:
                    with open(demo_dir, 'rb') as f:
                        demo = json.load(f)
                    # preprocess the demo data
                    # Log demo structure
                    log_data_utils(f"Demo type: {type(demo)}", "info")
                    if isinstance(demo, dict):
                        log_data_utils(f"Demo keys: {list(demo.keys())}", "info")
                        for key, value in demo.items():
                            if isinstance(value, np.ndarray):
                                log_data_utils(f"  {key}: shape={value.shape}, dtype={value.dtype}", "info")
                            else:
                                log_data_utils(f"  {key}: type={type(value)}, value={value}", "info")
                    elif isinstance(demo, list):
                        log_data_utils(f"Demo length: {len(demo)}", "info")
                        if len(demo) > 0:
                            log_data_utils(f"First item type: {type(demo[0])}", "info")
                            if isinstance(demo[0], dict):
                                log_data_utils(f"First item keys: {list(demo[0].keys())}", "info")
                except Exception as e:
                    demo = {}
                    log_data_utils(f"Failed to load demo data: {str(e)}", "error")

                # Set camera paths
                left_camera_dir = f"{root_dir}/{episode_number}/{self.left_camera_key}"
                right_camera_dir = f"{root_dir}/{episode_number}/{self.right_camera_key}"
                front_camera_dir = f"{root_dir}/{episode_number}/{self.front_camera_key}"

                log_data_utils(f"Looking for images in: {left_camera_dir}", "info")
                log_data_utils(f"Looking for images in: {right_camera_dir}", "info")
                log_data_utils(f"Looking for images in: {front_camera_dir}", "info")
                
                self.left_rgb_paths = sorted(glob.glob(os.path.join(left_camera_dir, "*.png")))
                self.right_rgb_paths = sorted(glob.glob(os.path.join(right_camera_dir, "*.png")))
                self.front_rgb_paths = sorted(glob.glob(os.path.join(front_camera_dir, "*.png")))

                log_data_utils(f"Found {len(self.left_rgb_paths)} left camera images", "info")
                log_data_utils(f"Found {len(self.right_rgb_paths)} right camera images", "info")
                log_data_utils(f"Found {len(self.front_rgb_paths)} front camera images", "info")

                if len(self.left_rgb_paths) == 0:
                    log_data_utils(f"WARNING: No left camera images found in {left_camera_dir}", "warning")
                if len(self.right_rgb_paths) == 0:
                    log_data_utils(f"WARNING: No right camera images found in {right_camera_dir}", "warning")
                if len(self.front_rgb_paths) == 0:
                    log_data_utils(f"WARNING: No front camera images found in {front_camera_dir}", "warning")
                
                def convert_to_int_list(value):
                    # Helper function to convert stringified lists to integer lists or string numbers to integers.
                    if isinstance(value, str):
                        # Check if the string is a stringified list
                        if value.startswith('[') and value.endswith(']'):
                            try:
                                # Try to evaluate the string as a list
                                result = ast.literal_eval(value)
                                if isinstance(result, list):
                                    # Convert all items of the list to integers if possible
                                    return [float(x) if isinstance(x, (int, float)) else 0 for x in result]
                                else:
                                    # Return the original value if it's not a list
                                    return value
                            except (ValueError, SyntaxError) as e:
                                # If we fail to parse the stringified list, log the error
                                log_data_utils(f"Error converting stringified list '{value}' to list: {e}", "warning")
                                return value  # Return the original value if conversion fails
                        else:
                            # If it's just a number in string format, convert it to an integer
                            try:
                                return int(value)
                            except ValueError:
                                # If it's not a valid number, return it as is
                                return value
                    return value

                resolved_demo = []
    
                for step_data in demo:
                    resolved_step = {}
                    
                    # Iterate over each key-value pair in the current step data
                    for key, value in step_data.items():
                        resolved_step[key] = convert_to_int_list(value)  # Apply conversion function to each value
                    
                    resolved_demo.append(resolved_step)

                self.demo = resolved_demo

                # Function to load and convert image to numpy array
                def load_image(path):
                    try:
                        return np.array(Image.open(path))
                    except Exception as e:
                        log_data_utils(f"Error processing image {path}: {str(e)}", "error")
                        return None

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Use map to load all images concurrently for each path
                    left_images = list(executor.map(load_image, self.left_rgb_paths))
                    right_images = list(executor.map(load_image, self.right_rgb_paths))
                    front_images = list(executor.map(load_image, self.front_rgb_paths))

                # Insert images into demo (ensure each image corresponds to the correct demo step)
                for i, demo_step in enumerate(self.demo):
                    if left_images[i] is not None:
                        demo_step["image_left_rgb"] = left_images[i]
                    if right_images[i] is not None:
                        demo_step["image_right_rgb"] = right_images[i]
                    if front_images[i] is not None:
                        demo_step["image_front_rgb"] = front_images[i]
                # self.demo[self.main_camera_key] = np.stack([np.array(Image.open(path)) for path in self.main_rgb_paths])
                # self.demo[self.wrist_camera_key] = np.stack([np.array(Image.open(path)) for path in self.wrist_rgb_paths])
                log_data_utils(f"Replaying episode from: {demo_dir}", "info")

                # Log comprehensive data information
                log_demo_data_info(self.demo[0], demo_dir)
            elif self.save_format == "npy":
                demo_dir = os.path.join(root_dir, self.save_format, f"{episode_number}.npy")
                with open(demo_dir, 'rb') as f:
                    self.demo = pickle.load(f)
                log_demo_data_info(self.demo, demo_dir)
        except Exception as e:
            log_data_utils(f"Error loading demo data: {str(e)}", "error")
            return False
        
        return True
    
    def get_demo_length(self):
        if self.demo is None:
            log_data_utils("No demo data loaded. Please load a demo first.", "error")
            return 0
        return len(self.demo)
    
    def get_observation(self, step_idx: int) -> Dict[str, Any]:
        """
        Format the observation dictionary.
        """
        # Create observation dictionary from demo data
        obs = {}
        for key in self.demo[step_idx].keys():
            obs[key] = self.demo[step_idx][key]
        return obs
    
    def get_instruction(self):
        if self.demo is None:
            log_data_utils("No demo data loaded. Please load a demo first.", "error")
            return None
        if "language_instruction" not in self.demo:
            log_data_utils("No language instruction found in demo.", "error")
            return None
        return self.demo["language_instruction"]
 
    def replay(self, env: RobotEnv, visual: bool = False, robot_trajectory: bool = True):
        if self.demo is None:
            log_data_utils("No demo data loaded. Please load a demo first.", "error")
            return
        
        # demo_length = self.demo["left_raw_action"].shape[0]
        demo_length = len(self.demo)
        joints = [np.concatenate([self.demo[i]["left_joint"], self.demo[i]["right_joint"]]) for i in range(demo_length)]
        # print(actions)
        input(f"Press Enter to replay the episode or Ctrl+C to exit...")
        
        log_data_utils(f"Starting replay of {demo_length} steps...", "info")
        log_data_utils(f"Demo keys: {list(self.demo[0].keys())}", "info")
        log_data_utils(f"Left camera key: {self.left_camera_key}", "info")
        log_data_utils(f"Right camera key: {self.right_camera_key}", "info")
        log_data_utils(f"Front camera key: {self.front_camera_key}", "info")

        if self.left_camera_key is not None:
            log_data_utils(f"Left camera images: {len(self.left_rgb_paths)}", "info")
        if self.right_camera_key is not None:
            log_data_utils(f"Right camera images: {len(self.right_rgb_paths)}", "info")
        if self.front_camera_key is not None:
            log_data_utils(f"Front camera images: {len(self.front_rgb_paths)}", "info")
        
        try:
            def move_to_start_position(
                env,
                target_joints: np.ndarray,
            ):
                curr_joints = env.get_obs()["joint_positions"]

                max_delta = (np.abs(curr_joints - target_joints)).max()
                steps = min(int(max_delta / 0.01), 100)
                print(f"Moving to start position with {steps} steps")

                # print(f"Moving robot to target joints position: {target_joints}")
                for jnt in np.linspace(curr_joints, target_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)

            if robot_trajectory:
                for step_idx in tqdm(range(demo_length), desc="Replaying episode"):
                    target_joints = joints[step_idx]
                    
                    # Log step information
                    if step_idx % 15 == 0:  # Log every 10 steps to avoid spam
                        log_data_utils(f"Step {step_idx}/{demo_length}: joints={target_joints[6]}, gripper={target_joints[6].tolist()}", "data_info")
                        log_data_utils(f"Step {step_idx}/{demo_length}: joints={target_joints[13]}, gripper={target_joints[13].tolist()}", "data_info")
                    
                    # # Visualize episode if requested
                    # if visual:
                    #     self.visualize_episode(obs, step_idx, act)
                    
                    # env.step(act)
                    obs = move_to_start_position(env, target_joints)
                    

                                # plt.waitforbuttonpress()
            
            # Initialize visualization if requested
            if visual:
                self._init_visualization()
                for step_idx in tqdm(range(demo_length), desc="Replaying episode"):
                    joint = joints[step_idx]

                    obs = self.get_observation(step_idx)
                    
                    # Log step information
                    if step_idx % 15 == 0:  # Log every 10 steps to avoid spam
                        log_data_utils(f"Step {step_idx}/{demo_length}: joints={joint[:6]}, gripper={joint[6]:.3f}", "data_info")
                        log_data_utils(f"Step {step_idx}/{demo_length}: joints={joint[7:13]}, gripper={joint[13]:.3f}", "data_info")
                    
                    # Visualize episode if requested
                    if visual:
                        self.visualize_episode(obs, step_idx)
                    
                    # env.step(act)
                    
                    # Handle window events for visualization
                    if visual:
                        plt.pause(0.001)  # Small pause to allow matplotlib to update
                        if plt.waitforbuttonpress(timeout=0.001):  # Check for key press
                            key = plt.gcf().canvas.get_key()
                            if key == 'q':  # Press 'q' to quit
                                log_data_utils("Replay interrupted by user (pressed 'q')", "warning")
                                break
                            elif key == 'p':  # Press 'p' to pause
                                log_data_utils("Replay paused. Press any key to continue...", "info")
                                plt.waitforbuttonpress()
                
                log_data_utils("Episode replay completed successfully!", "success")
            
        except Exception as e:
            log_data_utils(f"Error during replay: {str(e)}", "error")
            raise
        finally:
            # Clean up visualization
            if visual:
                self._cleanup_visualization()


    def visualize_episode(self, obs: Dict[str, Any], step_idx: int):
        """
        Main visualization function that orchestrates all visualization components.
        
        Args:
            obs: Observation dictionary containing image data and metadata
            step_idx: Current step index
            action: Current action being executed
        """
        # Update history for real-time plotting
        self.step_history.append(step_idx)
        
        # Visualize images
        self._visualize_image(obs)
        
        # Update the plot
        plt.tight_layout()
        plt.draw()

    def _init_visualization(self):
        """Initialize matplotlib figure and subplots for visualization."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout: 2 rows, 3 columns
        gs = self.fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1])
        
        # Image subplots (top row)
        self.ax_images = []
        for i in range(3):
            ax = self.fig.add_subplot(gs[0, i])
            self.ax_images.append(ax)

        self.ax_images[0].set_title("Left Camera")
        self.ax_images[1].set_title("Front Camera")
        self.ax_images[2].set_title("Right Camera")
        
        plt.tight_layout()

    def _cleanup_visualization(self):
        """Clean up matplotlib resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axs = None

    def _visualize_image(self, obs: Dict[str, Any]):
        """
        Visualize RGB and depth images using matplotlib.
        
        Args:
            obs: Observation dictionary containing image data
        """
        # Clear previous images
        for ax in self.ax_images:
            ax.clear()
        
        # Debug logging
        log_data_utils(f"Visualizing images for keys: {list(obs.keys())}", "debug")
        
        # Find image keys
        rgb_keys = [key for key in obs.keys() if 'rgb' in key.lower()]
        # depth_keys = [key for key in obs.keys() if 'depth' in key.lower()]
        
        log_data_utils(f"Found RGB keys: {rgb_keys}", "debug")
        # log_data_utils(f"Found depth keys: {depth_keys}", "debug")
        
        # Handle RGB images
        if rgb_keys:
            for i, key in enumerate(rgb_keys):
                if i < len(self.ax_images):
                    img_data = obs[key]
                    
                    # Skip if image is None
                    if img_data is None:
                        self.ax_images[i].text(0.5, 0.5, f"No {key} image", 
                                             transform=self.ax_images[i].transAxes,
                                             ha='center', va='center',
                                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                        self.ax_images[i].set_title(f"{key} (Not Available)")
                        self.ax_images[i].axis('off')
                        continue

                    # Convert PIL Image to numpy array if needed
                    if isinstance(img_data, Image.Image):
                        img_data = np.array(img_data)
                    
                    # Ensure img_data is a numpy array with proper dtype
                    if not isinstance(img_data, np.ndarray):
                        log_data_utils(f"Warning: img_data is not a numpy array, type: {type(img_data)}", "warning")
                        continue
                    
                    # Convert object dtype to proper numeric dtype if needed
                    if img_data.dtype == np.dtype('object'):
                        try:
                            img_data = img_data.astype(np.float32)
                        except:
                            log_data_utils(f"Warning: Could not convert image data to float32", "warning")
                            continue
                    
                    # Handle different image formats
                    if len(img_data.shape) == 3:
                        if img_data.shape[2] == 3:
                            # RGB image
                            self.ax_images[i].imshow(img_data)
                        elif img_data.shape[2] == 4:
                            # RGBA image, convert to RGB
                            self.ax_images[i].imshow(img_data[:, :, :3])
                    else:
                        # Grayscale image
                        self.ax_images[i].imshow(img_data, cmap='gray')
                    
                    self.ax_images[i].set_title(f"{key}")
                    self.ax_images[i].axis('off')
        
        for i in range(len(rgb_keys), len(self.ax_images)):
            self.ax_images[i].set_visible(False)


def run_replay_mode(cfg, env: RobotEnv, data_replayer: DataReplayer, logger_func=log_collect_demos):
    """
    Run interactive replay mode that can be used by both policy_eval.py and replay.py
    
    Args:
        cfg: Configuration object
        env: Robot environment
        data_replayer: DataReplayer instance
    """
    logger_func("Starting replay mode", "important")
    base_dir = cfg.storage.base_dir
    last_task_directory = None
    last_episode_number = None
    
    while True:
        try:
            env.reset()
            logger_func("Environment reset successfully", "success")

            # Get task directory
            if last_task_directory:
                task_directory = input(f"Enter the record_date/task_directory (e.g. date_723/debug) [last: {last_task_directory}]: ")
                if not task_directory.strip():
                    task_directory = last_task_directory
            else:
                task_directory = input("Enter the record_date/task_directory (e.g. date_723/debug): ")
            
            # Get episode number
            if last_episode_number:
                episode_number = input(f"Enter the episode number [last: {last_episode_number}]: ")
                if not episode_number.strip():
                    episode_number = last_episode_number
            else:
                episode_number = input("Enter the episode number: ")
            
            # Store for next iteration
            last_task_directory = task_directory
            last_episode_number = episode_number
            
            root_dir = f"{base_dir}/{task_directory}"

            # Ask user for visualization options
            visual_choice = input("Enable image visualization? (y/n): ").lower().strip()
            visual = visual_choice in ['y', 'yes']
            
            data_replayer.load_episode(root_dir, episode_number, main_camera_key=cfg.camera.main_camera_key, wrist_camera_key=cfg.camera.wrist_camera_key)
            data_replayer.replay(env, visual=visual)

            logger_func("Episode replay completed", "success")
            
            # Ask if user wants to repeat the same episode
            repeat = input("Repeat the same episode? (y/n): ").lower().strip()
            if repeat in ['y', 'yes']:
                logger_func("Repeating the same episode...", "info")
                continue
            elif repeat in ['n', 'no']:
                logger_func("Moving to next episode...", "info")
                continue
            else:
                logger_func("Invalid input. Moving to next episode...", "warning")
                continue
                
        except KeyboardInterrupt:
            logger_func("Replay interrupted by user. Exiting...", "info")
            break
        except Exception as e:
            logger_func(f"Error during replay: {str(e)}", "error")
            import traceback
            logger_func(f"Full traceback:\n{traceback.format_exc()}", "error")
            continue

def replay_episode_pickle(demo_dir, env: RobotEnv):
    with open(demo_dir, 'rb') as f:
        demo = pickle.load(f)
   
    actions = demo["action"]

    demo_length = actions.shape[0]
    for step_idx in tqdm(range(demo_length)):
        act = actions[step_idx]
        env.step(act)