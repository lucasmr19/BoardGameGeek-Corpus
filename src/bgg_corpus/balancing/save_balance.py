from datetime import datetime
import json
import os

from ..resources import REPORTS_DIR, LOGGER

def save_balance_report(stats, base_dir=REPORTS_DIR):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = os.path.join(base_dir, f"balance_report_{timestamp}.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    LOGGER.info(f"Balance report saved at: {report_path}")
    return report_path