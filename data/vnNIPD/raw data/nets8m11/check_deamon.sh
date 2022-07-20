#!/bin/bash

# this script checks if there are remaining active background processes
# it looks for any process which have calculate in it's name
# if the only output of this files is one line which ends with "grep calculate"
# then the are no remaining background processes

ps aux | grep calculate

