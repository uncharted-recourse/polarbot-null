#!/bin/sh

set -x
echo "Starting up response engine..."
python3 polarbot-respond.py -m $MESOS_SANDBOX/$MODEL_PATH \
                             -v $MESOS_SANDBOX/$VERSION_STR \
                             $NMT_FLAGS
