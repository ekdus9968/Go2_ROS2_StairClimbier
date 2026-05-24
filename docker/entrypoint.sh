#!/bin/bash
set -e

# X11 display for MuJoCo viewer (WSLg or X11)
if [ -e /tmp/.X11-unix ]; then
    export DISPLAY=${DISPLAY:-:0}
fi

# Print environment info
echo "=== Container started ==="
echo "Python: $(python --version)"
echo "PyTorch/JAX devices:"
python -c "import jax; print('JAX:', jax.devices())" 2>&1 | head -5 || true
echo ""

exec "$@"
