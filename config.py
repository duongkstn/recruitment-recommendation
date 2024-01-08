import os
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

for key in config:
    if isinstance(config[key], bool) and key in os.environ:
        if os.getenv(key) == "True":
            config[key] = True
        else:
            config[key] = False
    else:
        config[key] = os.getenv(key, config[key])
