from pathlib import Path

import platformdirs

APP_NAME = "rapmat-materials"

APP_TMPDIR_SUFFIX = "rapmatmaterials"

APP_CONFIG_DIR = Path(platformdirs.user_config_dir(APP_NAME))
APP_DATA_DIR = Path(platformdirs.user_data_dir(APP_NAME))
