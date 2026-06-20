import logging
import json
import time


# 1. Custom JSON Formatter Banaya
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_object = {
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime(record.created)
            ),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        # Agar koi extra data pass kiya ho toh woh bhi JSON mein add ho jaye
        if hasattr(record, "extra_data"):
            log_object["extra"] = record.extra_data

        return json.dumps(log_object)


# 2. Logger Configuration
logger = logging.getLogger("mlops_logger")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

if __name__ == "__main__":
    # Test Logs
    logger.info("Validation pipeline successfully initialized.")

    # Extra data ke sath log karna
    logger.error(
        "Invalid coordinate detected!",
        extra={"extra_data": {"file": "label1.txt", "line": 3}},
    )
