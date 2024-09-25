#!/bin/bash

# Loop from 1 to 2000 and run the Python script with each number as an argument
for i in {1..2000}
do
    python CalculatePowerForSingleOutcome.py $i
done