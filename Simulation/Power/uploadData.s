#!/bin/bash

tar -czvf ResultComplete.tar.gz ResultComplete

# Add the archive to git staging
git add ResultComplete.tar.gz

# Commit the change with a message
git commit -m "Data" 

# Push to the repository
git push

