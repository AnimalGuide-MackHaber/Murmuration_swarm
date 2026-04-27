import datetime
from typing import List, Dict

class SwarmLogger:
    _instance = None
    _logs: List[Dict[str, str]] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SwarmLogger, cls).__new__(cls)
        return cls._instance

    def log(self, message: str, level: str = "INFO", source: str = "System"):
        """Adds a log entry."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "level": level,
            "source": source,
            "message": message
        }
        self._logs.append(entry)
        # Keep buffer size manageable? 1000 lines?
        if len(self._logs) > 1000:
            self._logs.pop(0)
            
        print(f"[{timestamp}] [{level}] [{source}] {message}", flush=True)

    def get_logs(self) -> List[Dict[str, str]]:
        """Returns all logs."""
        return self._logs

    def clear(self):
        """Clears the log buffer."""
        self._logs = []
