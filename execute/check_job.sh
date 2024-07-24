#!/bin/bash

while true; do
    clear
    qstat -u s2210434
    sleep $1
done
