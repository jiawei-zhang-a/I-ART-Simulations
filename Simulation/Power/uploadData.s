#!/bin/bash

tar -czvf Result.tar.gz Result

# Add the archive to git staging
git add Result.tar.gz

# Commit the change with a message
git commit -m "Data" 

# Push to the repository
git push

