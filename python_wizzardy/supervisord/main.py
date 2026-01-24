import os
import time

def run():
    worker_id = os.getenv("WORKER_ID", "unknown")
    print(f"[worker {worker_id}] starting...", flush=True)

    while True:
        print(f"[worker {worker_id}] heartbeat", flush=True)
        time.sleep(5)

if __name__ == "__main__":
    run()
