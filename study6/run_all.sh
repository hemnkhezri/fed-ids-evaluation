#!/bin/bash
# Run from the study6 folder. partitions/dev must contain the Study 3 partition files (copy or link ../study3/partitions).
python code/run_s6.py --phase select --workers 2
python code/run_s6.py --phase curves --workers 2
python code/run_s6.py --phase confirm --workers 2
python code/analyze_s6.py
