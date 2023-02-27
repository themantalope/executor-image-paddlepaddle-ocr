#!/bin/bash
echo "Current working directory: $(pwd)"
echo "Current user: $(whoami)"
if [ -d "/PaddleOCR" ]; then
    echo "PaddleOCR directory exists"
else
    echo "PaddleOCR directory does not exist"
fi

mkdir logs
jina executor --uses config.yml --timeout-ready 3000000 > logs/jina.log 2>&1 &

until [ -f logs/jina.log ]
do
     sleep 1
done

cat logs/jina.log

nohup jupyter-lab --port 8888 --allow-root &


# /bin/bash
# exec "$@"