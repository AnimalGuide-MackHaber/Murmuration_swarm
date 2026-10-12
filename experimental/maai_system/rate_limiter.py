import asyncio
import time

class AsyncRateLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        
        # Requests token bucket
        self.req_capacity = requests_per_minute
        self.req_tokens = requests_per_minute
        self.req_fill_rate = requests_per_minute / 60.0
        
        # Tokens token bucket
        self.tok_capacity = tokens_per_minute
        self.tok_tokens = tokens_per_minute
        self.tok_fill_rate = tokens_per_minute / 60.0
        
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def _update_tokens(self):
        now = time.monotonic()
        elapsed = now - self.last_update
        
        # Refill req bucket
        self.req_tokens = min(self.req_capacity, self.req_tokens + elapsed * self.req_fill_rate)
        
        # Refill tok bucket
        self.tok_tokens = min(self.tok_capacity, self.tok_tokens + elapsed * self.tok_fill_rate)
        
        self.last_update = now

    async def wait_if_needed(self, estimated_tokens: int = 1000):
        async with self._lock:
            while True:
                await self._update_tokens()
                
                # Check if we have enough tokens in both buckets
                if self.req_tokens >= 1 and self.tok_tokens >= estimated_tokens:
                    self.req_tokens -= 1
                    self.tok_tokens -= estimated_tokens
                    return
                
                # Calculate time to wait based on which bucket is empty
                wait_time_req = 0.0
                wait_time_tok = 0.0
                
                if self.req_tokens < 1:
                    wait_time_req = (1 - self.req_tokens) / self.req_fill_rate
                
                if self.tok_tokens < estimated_tokens:
                    wait_time_tok = (estimated_tokens - self.tok_tokens) / self.tok_fill_rate
                
                sleep_time = max(wait_time_req, wait_time_tok)
                
                # Ensure we don't sleep forever on edge cases, add tiny buffer
                sleep_time = max(0.1, sleep_time)
                
                # print(f"[RateLimiter] Throttling... Sleeping for {sleep_time:.2f}s (Req bucket: {self.req_tokens:.1f}, Tok bucket: {self.tok_tokens:.1f})")
                
                # Yield control to event loop while sleeping
                # We need to sleep outside the lock if we don't want to block other wait calls completely?
                # Actually, blocking inside the lock ensures FIFO queuing for the tokens, which is usually desired.
                # However, asyncio.sleep inside lock prevents other tasks from updating. 
                # It's better to calculate sleep time, release lock, sleep, and try again.
                break

        # If we broke out of the loop to sleep
        await asyncio.sleep(sleep_time)
        return await self.wait_if_needed(estimated_tokens)

    async def refund_tokens(self, tokens_to_refund: int):
        """Refund tokens if the estimation was too high."""
        async with self._lock:
            await self._update_tokens()
            self.tok_tokens = min(self.tok_capacity, self.tok_tokens + tokens_to_refund)
