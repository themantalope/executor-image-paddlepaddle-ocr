#!/bin/bash
echo "Current working directory: $(pwd)"
echo "Current user: $(whoami)"
if [ -d "/PaddleOCR" ]; then
    echo "PaddleOCR directory exists"
else
    echo "PaddleOCR directory does not exist"
fi

mkdir logs
jina executor --uses config.yml > logs/jina.log 2>&1 &
# exec "$@"