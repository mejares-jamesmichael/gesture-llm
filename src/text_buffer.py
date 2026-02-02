import time


class TextBuffer:
    def __init__(self, timeout=3.0):
        self.buffer = []
        self.last_update = time.time()
        self.timeout = timeout
    
    def add_word(self, word):
        self.buffer.append(word)
        self.last_update = time.time()
    
    def get_sentence(self):
        return ' '.join(self.buffer)
    
    def should_send(self):
        return (time.time() - self.last_update) > self.timeout and self.buffer
    
    def clear(self):
        self.buffer = []
        self.last_update = time.time()
    
    def get_timeout_remaining(self):
        elapsed = time.time() - self.last_update
        remaining = max(0, self.timeout - elapsed)
        return remaining
