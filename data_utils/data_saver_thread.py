import threading
import queue

from data_utils.data_saver import DataSaver

class EpisodeSaverThread(threading.Thread):
    """
    A background thread that listens for completed episodes and saves them asynchronously.
    """
    def __init__(self, data_saver: DataSaver):
        super().__init__()
        self.data_saver = data_saver
        self.episode_queue = queue.Queue()  # Queue to hold episodes to save
        self.daemon = True  # Ensure the thread exits when the main program ends

    def run(self):
        """
        Continuously listens for episodes to save.
        """
        while True:
            try:
                # Wait for a new episode to save
                episode_data = self.episode_queue.get(timeout=5)  # Wait for 5 seconds
                if episode_data is None:  # Exit signal
                    break
                # self.data_saver.buffer = episode_data
                self.data_saver.save_episode_json(episode_data, pickle_only=False)
                self.episode_queue.task_done()  # Mark the task as done
            except queue.Empty:
                continue

    def save_episode(self, episode_data):
        """
        Put episode data in the queue for background saving.
        """
        self.episode_queue.put(episode_data)

    def stop(self):
        """Signal to stop the background thread."""
        self.episode_queue.put(None)
