from collections import deque
import numpy as np
from random import sample

class ExperienceReplay(object):

    def __init__(self, size=1000):
        """ Initialize memory with maximum size and empty buffer """
        self.max_size = size
        self.current_size = 0
        self.buffer = deque()

    def reset(self):
        """ Clear buffer, size = 0 """
        self.buffer.clear()
        self.current_size = 0

    def remember(self, sarsd_tuple):
        """
        Append one experience (sarsd_tuple) to buffer.
        If buffer is full, remove first element first.
        """
        if self.current_size < self.max_size:
            self.current_size += 1
        else:
            self.buffer.popleft()
        self.buffer.append(sarsd_tuple)

    def get_batch(self, batch_size):
        """
        Return a random batch of experiences of size batch_size from buffer.
        If less samples than batch_size available,
        sample as many experiences as possible,
        else sample batch_size experiences from buffer
        """
        if self.current_size < batch_size:
            return sample(self.buffer, self.current_size)
        else:
            return sample(self.buffer, batch_size)
