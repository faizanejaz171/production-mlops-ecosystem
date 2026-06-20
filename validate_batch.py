import os
import shutil
import logging
import json
import time
import requests
from dotenv import load_dotenv
from validate_schema import YOLOLabel

# Load environment variables from .env file for secure access keys
load_dotenv()


# --- LOGGER SETUP START ---
class JSONFormatter(logging.Formatter):
    """Custom formatter to output logs in structured production-grade JSON format."""

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


def send_slack_summary(summary, folder_name):
    """Transmits the final verification metrics to the configured Slack channel."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning(
            "Slack webhook URL is missing from environment variables. Skipping alert."
        )
        return

    # Construct the payload card layout for Slack
    slack_payload = {
        "text": f"🚨 *MLOps CV Pipeline Validation Report* 🚨\n"
        f"• *Target Directory:* `{folder_name}`\n"
        f"• *Total Files Processed:* {summary['total_files']}\n"
        f"• *Passed (Routed to Clean):* ✅ {summary['passed']}\n"
        f"• *Failed (Routed to Rejected):* ❌ {summary['failed']}\n"
        f"• *Top Anomalies:* `{json.dumps(summary['top_errors'])}`"
    }

    try:
        response = requests.post(webhook_url, json=slack_payload, timeout=10)
        if response.status_code == 200:
            logger.info("Validation summary metrics successfully transmitted to Slack.")
        else:
            logger.error(
                f"Slack API returned non-200 configuration status: {response.status_code}"
            )
    except Exception as e:
        logger.error(f"Network error encountered while reaching Slack Webhook: {e}")


def validate_batch(data_folder):
    """Scans, validates, and routes YOLO annotation text files based on data integrity."""
    # Define production routing directories
    clean_dir = "clean_dataset"
    rejected_dir = "rejected_dataset"

    # Automatically create directory targets if they do not exist
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

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
                    # Skip completely empty or blank lines
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

                        # ─── ADVANCED ANOMALY CHECK ───
                        # Reject bounding boxes that have zero or physically impossible areas
                        if data["width"] * data["height"] < 0.0001:
                            raise ValueError(
                                "Bounding box area is abnormally tiny or zero (Bad Annotation)."
                            )

                        YOLOLabel(**data)
                    except Exception as e:
                        file_has_error = True
                        error_msg = str(e).replace("\n", " ")

                        # Detailed ERROR log for backend analytics / Datadog aggregation
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
                        failure_reasons[error_msg] = (
                            failure_reasons.get(error_msg, 0) + 1
                        )

            # ─── MECHANICAL DATA ROUTING LOGIC ───
            if file_has_error:
                failed_files += 1
                shutil.copy(filepath, os.path.join(rejected_dir, filename))
            else:
                passed_files += 1
                shutil.copy(filepath, os.path.join(clean_dir, filename))
                logger.info(
                    "File passed validation completely",
                    extra={"extra_data": {"file": filename}},
                )

    # Compile the telemetry metrics structure
    summary_report = {
        "total_files": total_files,
        "passed": passed_files,
        "failed": failed_files,
        "top_errors": failure_reasons,
    }

    # Structured JSON Log output for system streaming
    logger.info(
        "Batch validation complete. Final Summary Report.",
        extra={"extra_data": summary_report},
    )

    # ─── HUMAN-READABLE TERMINAL BOX GENERATION ────────────────────────────
    print("\n" + "═" * 60)
    print(" 📊 DATA VALIDATION & ROUTING REPORT (SUMMARY)")
    print("═" * 60)
    print(f" 📁 Target Directory  : {data_folder}")
    print(f" Total Files Scanned  : {total_files}")
    print(f" Passed & Routed Clean: ✅ {passed_files} -> [{clean_dir}/]")
    print(f" Failed & Segregated  : ❌ {failed_files} -> [{rejected_dir}/]")

    if failure_reasons:
        print("─" * 60)
        print(" ⚠️  Top Failure Breakdown:")
        for reason, count in failure_reasons.items():
            print(f"   • {reason[:75]}... : {count} occurrence(s)")

    print("═" * 60 + "\n")
    # ───────────────────────────────────────────────────────────────────────

    # Dispatch metrics report via Slack webhook integration
    send_slack_summary(summary_report, data_folder)


if __name__ == "__main__":
    import argparse

    # Initialize the argument parser for professional command-line interface (CLI)
    parser = argparse.ArgumentParser(
        description="Production-grade MLOps YOLO Dataset Automation & Validation Tool."
    )

    # Add optional argument for folder path with a default value
    parser.add_argument(
        "--folder",
        type=str,
        default="sample_data",
        help="Path to the directory containing YOLO format .txt label files. (Default: sample_data)",
    )

    # Parse incoming CLI arguments safely
    args = (
        parser.parse_parser_args()
        if hasattr(parser, "parse_parser_args")
        else parser.parse_args()
    )

    # Execute the batch validation workflow with the dynamically provided folder path
    validate_batch(args.folder)
