# Source from the package root: data root, import path and the attention backend note.
export VEGO_DATA=${VEGO_DATA:-$(pwd)/vego_data}
export PYTHONPATH=$(pwd)${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=${HF_HOME:-$VEGO_DATA/cache/hf}
