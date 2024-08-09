#!/bin/bash

# get is yaml config file is passed
if [ -z "$1" ]; then
    echo "Usage: $0 no config file provided"
    exit 1
fi

CONFIG_FILE="$1"

#run script
python3 plot_driver.py --config_file "$CONFIG_FILE"