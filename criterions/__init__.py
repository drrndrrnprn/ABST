import importlib
import os


# ASG loss requires flashlight bindings
files_to_skip = set()

for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_") and file not in files_to_skip:
        criterion_name = file[: file.find(".py")]
        importlib.import_module(
            "..bertabst.criterions." + criterion_name
        )
