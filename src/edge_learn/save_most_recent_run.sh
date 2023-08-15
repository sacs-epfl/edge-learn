#!/bin/bash

BASE_DIR=./results
DEST_DIR=~/results

MOST_RECENT_DIR=$(ls -td -- "$BASE_DIR"/*/ | head -n 1)
MOST_RECENT_DIR=${MOST_RECENT_DIR%/}

echo "Saving images from ${MOST_RECENT_DIR}"
cp $MOST_RECENT_DIR/edge_server/*.png ~/results
