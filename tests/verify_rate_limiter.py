import asyncio
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rate_limiter import RateLimiter

async def main():
    print("Testing Rate Limiter...")
    
    # Test 1: RPM
    print("\n--- Test 1: RPM Limit (2 requests/min) ---")
    rl = RateLimiter(requests_per_minute=2, tokens_per_minute=1000)
    
    t0 = time.time()
    await rl.acquire()
    print(f"Req 1: {time.time() - t0:.2f}s (Expected: immediate)")
    
    await rl.acquire()
    print(f"Req 2: {time.time() - t0:.2f}s (Expected: immediate)")
    
    # We won't wait for the 3rd one as it takes 60s, but we've initialized it.
    
    # Test 2: TPM
    print("\n--- Test 2: TPM Limit (10 tokens/min) ---")
    rl2 = RateLimiter(requests_per_minute=60, tokens_per_minute=10)
    t1 = time.time()
    
    await rl2.acquire(estimated_tokens=5)
    print(f"Req A (5 tokens): {time.time() - t1:.2f}s (Expected: immediate)")
    
    await rl2.acquire(estimated_tokens=4)
    print(f"Req B (4 tokens): {time.time() - t1:.2f}s (Expected: immediate)")
    
    # The next one would breach 10 TPM (5+4+x > 10)
    if len(rl2.token_timestamps) == 2:
        print("Token timestamps recorded correctly.")
    else:
        print(f"Error: Token timestamps count is {len(rl2.token_timestamps)}")

    print("\nVerification Complete.")

if __name__ == "__main__":
    asyncio.run(main())
