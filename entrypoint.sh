echo "Current working directory: $(pwd)"
echo "Current user: $(whoami)"
if [ -d "/PaddleOCR" ];
    echo "PaddleOCR directory exists"
else 
    echo "PaddleOCR directory does not exist"
fi

# jina executor --uses config.yml