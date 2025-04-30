#!/bin/bash
pip install -r requirements.txt 

# Install PySvelte:
pip install git+https://github.com/Mech-Interp/PySvelte.git
# Needed for PySvelte to work
pip install typeguard==2.13.3
pip install typing-extensions

# Fix ERR_OSSL_EVP_UNSUPPORTED when importing PySvelte 
export NODE_OPTIONS=--openssl-legacy-provider
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
# Navigate to the PySvelte build directory
cd ~/.conda/envs/RoPE/lib/python${PYVER}/site-packages/pysvelte/svelte
# Rebuild
npm run webpack