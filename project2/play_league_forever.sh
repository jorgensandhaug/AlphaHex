#!/bin/bash

# Initialize a counter
counter=0

# File to store the log output
log_file="play_online_log.txt"

# Infinite loop to run the command continuously
while true; do
    # Increment the counter
    ((counter++))

    # Log start time and count
    echo "Run #$counter started at $(date)" >> $log_file
    
    # Execute the command
    python3 play_online.py
    
    # Log end time
    echo "Run #$counter finished at $(date)" >> $log_file
    echo "--------------------------------" >> $log_file

    # Optional: sleep for a second (or any desired duration) to pace the executions
    # sleep 1
done

