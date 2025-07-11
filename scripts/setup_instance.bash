#!/usr/bin/env bash
set -e

# 1. Install Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v yum &> /dev/null; then
            # CentOS/RockyOS
            sudo yum install -y gcc gcc-c++ make
            sudo yum install -y python3.11 python3.11-venv python3.11-distutils
        else
            # Ubuntu/Debian
            sudo apt-get update
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt-get update
            sudo apt-get install -y python3.11 python3.11-venv python3.11-distutils
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install python@3.11
    else
        echo "Unsupported OS. Please install Python 3.11 manually."
        exit 1
    fi
else
    echo "Python 3.11 already installed."
fi

# 2. Install Poetry
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3.11 -
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Poetry already installed."
fi

# 3. Run poetry install
echo "Running 'poetry install'..."
poetry install

# 4. Clone the Govscape repository
git clone https://github.com/bcglee/govscape

cd govscape

# 5. Check for arguments and execute corresponding scripts
if [[ "$1" == "--server" ]]; then
    echo "Running server setup..."
    poetry run python download_s3_embeddings.py
    poetry run python start_api_server.py
elif [[ "$1" == "--embeddings" ]]; then
    echo "Running embeddings pipeline..."
    poetry run python s3_ec2_embedding_pipeline.py
else
    echo "Invalid argument. Use '--server' or '--embeddings'."
    exit 1
fi
