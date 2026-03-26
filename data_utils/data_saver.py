import concurrent
import json
import logging
import os
from re import L
import shutil
import time
from PIL import Image

logger = logging.getLogger("data_saver")
logger.setLevel(logging.INFO)


class DataSaver:
    def __init__(
        self,
        save_dir="/home/sean/Desktop/YAM/gello_software/data",
        task_directory="Testing_dir",
        language_instruction="Test",
        saver_max_workers=None,
        png_compress_level=1,
    ):
        self.save_dir = os.path.join(
            save_dir,
            task_directory
        )

        # self.task = task
        self.traj_count = 1 # number of actions saved
        self.buffer = [] # buffer for a single action
        self.instruction = language_instruction
        # Limit workers to avoid CPU spikes from the saver.
        if saver_max_workers is None:
            self.max_workers = max(1, min(4, (os.cpu_count() or 1)))
        else:
            self.max_workers = max(1, int(saver_max_workers))
        # Lower PNG compression is much faster (larger files, lower CPU).
        self.png_compress_level = max(0, min(9, int(png_compress_level)))

        if os.path.exists(self.save_dir):
            remove_dir = input(f"The directory {self.save_dir} already exists. Do you want to remove it? (y/n): ")
            if remove_dir == "y":
                shutil.rmtree(self.save_dir)
                logger.info(f"Removed existing directory: {self.save_dir}.")
            elif remove_dir == "n":
                append_dir = input(f"Do you want to append to the existing directory? (y/n): ")
                if append_dir == "y":
                    self.traj_count = int(input(f"Enter the next episode number to append to the directory: "))
                    logger.info(f"Appending to existing directory: {self.save_dir} starting with episode number {self.traj_count}.")
                else:
                    raise FileExistsError(f"The directory {self.save_dir} already exists.")
            else:
                raise FileExistsError(f"The directory {self.save_dir} already exists.")

        os.makedirs(self.save_dir, exist_ok=True)
    
    def reset_buffer(self):
        old_size = len(self.buffer)
        self.buffer = []
        logger.info(f"Reset buffer: {old_size} observations cleared.")
        
    def add_observation(self, obs):
        obs_copy = {}
        obs_copy['instruction'] = self.instruction
        obs_copy['left_rgb'] = obs['left_camera_rgb']
        obs_copy['right_rgb'] = obs['right_camera_rgb']
        obs_copy['front_rgb'] = obs['front_camera_rgb']
        obs_copy['joint'] = obs['joint_positions']
        obs_copy['next_joint'] = obs['next_joint']
        self.buffer.append(obs_copy)

    def save_episode_json(self, buffer, pickle_only=False):
        if not self.buffer:
            logger.warning("Empty buffer, no observations to save.")

        logger.info(f"Saving episode {self.traj_count} to {self.save_dir} with {len(buffer)} observations.")
        if buffer == []:
            logger.warning("Empty buffer, no observations to save.")
            return

        img_paths = {}
        task_name = self.instruction
        joints = [obs["joint"] for obs in buffer]
        next_joints = [obs["next_joint"] for obs in buffer]

        if not pickle_only:
            # save rgb from camera
            rgb_keys = [key for key in buffer[0].keys() if "rgb" in key]
            rgb_keys.sort()  # Sort from smallest to largest
            # logger.info(f"Found {len(rgb_keys)} RGB cameras: {rgb_keys}")
            
            for key in rgb_keys:
                save_dir = os.path.join(self.save_dir, f'{self.traj_count:06d}')
                os.makedirs(save_dir, exist_ok=True)
                
                save_dir = os.path.join(save_dir, key)
                
                os.makedirs(save_dir, exist_ok=True)
                # logger.info(f"Created directory for camera {key}: {save_dir}")

                paths = []
                tasks = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    for i, obs in enumerate(buffer):
                        img = obs[key]
                        img_path = os.path.join(save_dir, f'{i:06d}.png')
                        tasks.append(executor.submit(self.save_image, img, img_path))
                        paths.append(img_path)

                    # Wait for all parallel tasks to complete
                    concurrent.futures.wait(tasks)
                    img_paths.setdefault(key, []).extend(paths)
                    # logger.info(f"Saved {len(paths)} images for camera {key}")

            # add action and delta to json
            # note that raw action is the joint returned by the viser ik, while the other joint is from robot obs
            json_data = []
            for i in range(len(joints)):
                json_data_obs = {
                    'language_instruction': task_name,
                    'left_joint': str(joints[i][:7].tolist()),
                    'right_joint': str(joints[i][7:].tolist()),
                    'next_left_joint': str(next_joints[i][:7].tolist()),
                    'next_right_joint': str(next_joints[i][7:].tolist())
                }

                # add image paths to json
                for j, rgb_key in enumerate(rgb_keys):
                    json_data_obs[f"image_{rgb_key}"] = img_paths[rgb_key][i]
                
                json_data.append(json_data_obs)

            json_save_path = os.path.join(self.save_dir, f'{self.traj_count:06d}', f'{self.traj_count:06d}.json')
            os.makedirs(os.path.dirname(json_save_path), exist_ok=True)

            if not os.path.exists(json_save_path):
                with open(json_save_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
                # logger.info(f"Created new JSON file: with {len(json_data)} observations.")
            else:
                with open(json_save_path, 'r') as f:
                    existing_data = json.load(f)
                existing_data.extend(json_data)
                with open(json_save_path, 'w') as f:
                    json.dump(existing_data, f, indent=4)
                logger.info(f"Added {len(json_data)} observations to {json_save_path}")
        # Just to let user know that the episode is saved
        logger.info(f"Complete!!!! Saved episode {self.traj_count} to {self.save_dir} with {len(buffer)} observations.")
        self.traj_count += 1
    
    def save_image(self, image, path):
        Image.fromarray(image).save(path, compress_level=self.png_compress_level)
