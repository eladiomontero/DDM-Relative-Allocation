#!/bin/bash
# Erasing all the file created by program
# It should be used with care, otherwise the data from the experiment could be lost

rm *score *lock *history *move *~ *timestart *timeend *playedby *loggedin *ready
rm counterplayed firstround numberofusers round roundsnumber
rm countercalculated  counterready started totaltotal
rm PrisonersNet
rm M N
rm deamonstarted allloggedin
python erase.py