import asyncio
import time
from collections import deque
from typing import Optional

class RateLimiter:
    """
    Manages rate limiting for API calls (Requests Per Minute and Tokens Per Minute).
    Uses a sliding window approach.
    """
    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 32000):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        
        self.request_timestamps = deque()
        self.token_timestamps = deque() # Stores (timestamp, token_count) tuples
        
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 0):
        """
        Waits until a request can be made and sufficient tokens are available.
        Ensures strict RPM and TPM limits using a sliding window.
        """
        while True:
            async with self._lock:
                now = time.time()
                self._cleanup(now)
                
                # Check RPM
                if len(self.request_timestamps) >= self.rpm_limit:
                    wait_time = 60 - (now - self.request_timestamps[0])
                    if wait_time > 0:
                        # print(f"[RateLimiter] RPM limit hit. Waiting {wait_time:.2f}s...")
                        should_wait = True
                        calc_wait = wait_time + 0.1 # Buffer
                    else:
                        should_wait = False
                
                # Check TPM
                elif len(self.token_timestamps) > 0:
                    current_tokens = sum(count for _, count in self.token_timestamps)
                    if current_tokens + estimated_tokens > self.tpm_limit:
                        # Wait for oldest token batch to expire
                        wait_time = 60 - (now - self.token_timestamps[0][0])
                        if wait_time > 0:
                            # print(f"[RateLimiter] TPM limit hit ({current_tokens}/{self.tpm_limit}). Waiting {wait_time:.2f}s...")
                            should_wait = True
                            calc_wait = wait_time + 0.1
                        else:
                            should_wait = False
                    else:
                        should_wait = False
                else:
                    should_wait = False

                if not should_wait:
                    # Grant access
                    self.request_timestamps.append(now)
                    if estimated_tokens > 0:
                        self.token_timestamps.append((now, estimated_tokens))
                    return

            # Release lock before sleeping
            if should_wait:
                await asyncio.sleep(calc_wait)

    def _cleanup(self, now: float):
        """Removes timestamps older than 60 seconds."""
        while self.request_timestamps and now - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
            
        while self.token_timestamps and now - self.token_timestamps[0][0] > 60:
            self.token_timestamps.popleft()

    async def update_usage(self, actual_tokens: int):
        """
        Updates the token usage with the actual count after a request completes.
        This is useful if we reserved 0 or an estimate, and now know the real cost.
        """
        async with self._lock:
             now = time.time()
             # We just append the usage. If we "reserved" tokens earlier, 
             # passing `estimated_tokens` to acquire handles the reservation.
             # If `acquire` was called with 0 (or low estimate) and we just want to track actuals:
             self.token_timestamps.append((now, actual_tokens))
             self._cleanup(now)
