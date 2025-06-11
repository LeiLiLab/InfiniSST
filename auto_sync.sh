#!/bin/bash

WATCH_DIR=./
REMOTE_PATH=jiaxuanluo@taurus.cs.ucsb.edu:/home/jiaxuanluo/InfiniSST/

echo "📡 Watching $WATCH_DIR for .py and .sh changes..."

fswatch -r \
  --exclude=".*\.swp$" \
  --exclude=".*~$" \
  --exclude=".*/\.idea/.*" \
  --exclude=".*/\.git/.*" \
  --exclude=".*\.pyc$" \
  "$WATCH_DIR" | while read file
do
    echo "🔁 Detected change: $file"
    rsync -az -e "ssh -i ~/.ssh/id_rsa_lab" \
      --include='*/' \
      --include='*.py' \
      --include='*.sh' \
      --exclude='*' \
      "$WATCH_DIR/" "$REMOTE_PATH"
done