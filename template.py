import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name="Test_red"

list_of_files=[
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/uploads/__init__.py",
    f"src/{project_name}/uploads/csv",
    f"src/{project_name}/uploads/sheets",
    f"src/{project_name}/app_backend/__init__.py",
    f"src/{project_name}/app_backend/ingest.py",
    f"src/{project_name}/app_backend/insight_engine.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    
    else:
        logging.info(f"{filename} is already exists")