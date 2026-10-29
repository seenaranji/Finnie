"""Step 0 demo: print the config (secrets masked), log a line, time the startup.

Run from the project root:  .venv/bin/python -m scripts.show_config
"""
import json
import time

t0 = time.perf_counter()
from src.core.config import load_settings  # importing pulls in pydantic, yaml, ...
from src.core.logging import get_logger, setup_logging
t_import = time.perf_counter() - t0

t1 = time.perf_counter()
settings = load_settings()
t_load = time.perf_counter() - t1

setup_logging(settings.app.log_level)
log = get_logger("show_config")

print(json.dumps(settings.masked(), indent=2))
log.info("config loaded (log_level=%s)", settings.app.log_level)
log.info("startup: import=%.1f ms, load=%.1f ms, total=%.1f ms",
         t_import * 1000, t_load * 1000, (t_import + t_load) * 1000)
