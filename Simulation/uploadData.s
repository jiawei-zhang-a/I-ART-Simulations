#!/bin/bash

tar -czvf ResultMultiple.tar.gz ResultMultiple

# Add the archive to git staging
git add ResultMultiple.tar.gz

# Commit the change with a message
git commit -m "Data" 

# Push to the repository
git push

