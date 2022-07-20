#!/bin/bash

# this program stops any active program whose name is calculate
# it's purpose is to stop any remaining active background processes which are left from previous run
# or to stop the experiment if needed

kill `ps aux | grep calculate | awk '{print $2}'`

