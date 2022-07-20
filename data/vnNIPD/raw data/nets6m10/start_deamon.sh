#!/bin/bash

# Starting background program which controls the experiment
# It uses 6 parametars if it is run with wrong number of parametars it prints the usage
# The payoffs of Prisoner dilemma game are defined here and can be changed in this function if needed

Y=11 # number of parameters
N=$1
M=$2
Rmin=$3
Rmax=$4
mt_python=$5
file=$6
T=$7
R=$8
P=$9
S=${10}
point=${11}

if [ $# -lt $Y ] 
  then
    echo ""
    echo " usage: ./start_deamon.sh N M Rmin Rmax mt_python backup T R P S point "
    echo ""
    echo "N, M          - the dimentions of the lattice"
    echo "Rmin          - minimal round number in each step"
    echo "Rmax          - maximal round number in each step" 
    echo "mt_python     - time before daemon plays instead of player, recommend mt_python=40" 
    echo "backup        - name of back up file"
    echo "T             - Temptation to defect, default = 10"
    echo "R             - Reward for mutual cooperation, default = 7"
    echo "P             - Punishment for mutual defection, default = 0"
    echo "S             - Sucker's payoff, default = 0"
    echo "point         - The value of point"
  else
#    ./backup.sh $file 						# backing up the old results

    # starting first part of experiment
    cd software/data
    python makenet.py $N $M $Rmin $Rmax                         # making network of size NxM: files with neighbors for each player: usuario*
    								# and generate number of round between Rmin and Rmax 
    python makefiles.py    					# makes necessary files
    python calculate.py $mt_python $T $R $P $S $point > log &          # starting background program
    chmod a+w *                                                 # makes all files writable
    cd ..
    cd ..

fi
