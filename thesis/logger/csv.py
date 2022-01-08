"""TODO."""
import csv
import os
from typing import Any, Dict, List

from thesis.logger.abc import ABCLogger
from thesis.logger.std import PrinterLogger


class CsvLogger(PrinterLogger, ABCLogger):
    """TODO."""

    data: Dict[str, Dict[str, List[Any]]]

    def __init__(
        self,
        names_and_headers: Dict[str, List[str]],
        name: str = "runs",
        override=False,
    ):
        """TODO."""
        super().__init__(name)
        self.data = {
            fname: {h: [] for h in headers + ["idx"]}
            for fname, headers in names_and_headers.items()
        }
        for fname, content in self.data.items():
            if override or not os.path.exists(f"{name}/{fname}"):
                print("write")
                with open(f"{name}/{fname}.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(content.keys())

    def add_scalars(self, who, what, when):
        """TODO."""
        for header in self.data[who]:
            self.data[who][header].append(what[header] if header != "idx" else when)

    def flush(self):
        for fname, data in self.data.items():
            with open(f"{self.name}/{fname}.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerows(zip(*data.values()))
        self.data = {
            fname: {h: [] for h in headers} for fname, headers in self.data.items()
        }

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """TODO."""
        self.flush()
        return super().__exit__(exc_type, exc_value, exc_traceback)
