import os
import logging
import json
import time
from validate_schema import YOLOLabel


# --- LOGGER SETUP START ---
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
        if hasattr(record, "extra_data"):
            log_object["extra"] = record.extra_data
        return json.dumps(log_object)


logger = logging.getLogger("mlops_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
# --- LOGGER SETUP END ---


def validate_batch(data_folder):
    total_files = 0
    passed_files = 0
    failed_files = 0
    failure_reasons = {}

    logger.info(f"Starting batch validation on folder: '{data_folder}'")

    if not os.path.exists(data_folder):
        logger.error(f"Folder '{data_folder}' does not exist!")
        return

    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            total_files += 1
            filepath = os.path.join(data_folder, filename)
            file_has_error = False

            with open(filepath, "r") as f:
                for i, line in enumerate(f):
                    # Khali lines ko ignore karne ke liye
                    if not line.strip():
                        continue

                    try:
                        parts = list(map(float, line.split()))
                        data = {
                            "class_id": int(parts[0]),
                            "x_center": parts[1],
                            "y_center": parts[2],
                            "width": parts[3],
                            "height": parts[4],
                        }
                        YOLOLabel(**data)
                    except Exception as e:
                        file_has_error = True
                        error_msg = str(e).replace("\n", " ")
                        # Detailed ERROR log with extra context
                        logger.error(
                            "Validation failed inside file",
                            extra={
                                "extra_data": {
                                    "file": filename,
                                    "line": i + 1,
                                    "reason": error_msg,
                                }
                            },
                        )
                        # Summary ke liye reason save karna
                        failure_reasons[error_msg] = (
                            failure_reasons.get(error_msg, 0) + 1
                        )

            if file_has_error:
                failed_files += 1
            else:
                passed_files += 1
                logger.info(
                    "File passed validation completely",
                    extra={"extra_data": {"file": filename}},
                )

    # Final Execution Summary Log
    logger.info(
        "Batch validation complete. Final Summary Report.",
        extra={
            "extra_data": {
                "total_files": total_files,
                "passed": passed_files,
                "failed": failed_files,
                "top_errors": failure_reasons,
            }
        },
    )


if __name__ == "__main__":
    validate_batch("sample_data")
