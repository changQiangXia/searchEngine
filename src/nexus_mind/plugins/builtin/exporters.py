"""Built-in exporter plugins."""

import csv
import json
from typing import Any

from nexus_mind.plugins.base import ExporterPlugin, PluginInfo


class CSVExporter(ExporterPlugin):
    """Export results to CSV."""

    info = PluginInfo(
        name="csv_exporter",
        version="1.0.0",
        description="Export search results to CSV",
        author="NexusMind",
    )

    def initialize(self) -> bool:
        return True

    def export(self, results: list[dict[str, Any]], output_path: str) -> bool:
        """Export to CSV."""
        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                if not results:
                    return True

                fieldnames = ["rank", "score"] + list(results[0].get("metadata", {}).keys())

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for r in results:
                    row = {
                        "rank": r.get("rank", 0),
                        "score": r.get("score", 0),
                        **r.get("metadata", {}),
                    }
                    writer.writerow(row)

            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def supported_formats(self) -> list[str]:
        return [".csv"]


class JSONExporter(ExporterPlugin):
    """Export results to JSON."""

    info = PluginInfo(
        name="json_exporter",
        version="1.0.0",
        description="Export search results to JSON",
        author="NexusMind",
    )

    def initialize(self) -> bool:
        return True

    def export(self, results: list[dict[str, Any]], output_path: str) -> bool:
        """Export to JSON."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def supported_formats(self) -> list[str]:
        return [".json"]
