#!/bin/bash

# This function is making the backup of experimental data.
# It copies the folders exp1/data, control/data, exp2/data 
# to the folders ./backups/name/exp1, ./backup/name/control, ./backup/name/exp2
# where name is the name of back up folder which is given as parametar of this script 

Y=1   		# number of parametars
name=$1 	# name of the backup file

if [ $# -lt $Y ]
  then
      echo "usage: ./backup.sh name"
      echo "name is name of the backup file"
  else

cd backups
mkdir $1
cd $1
mkdir exp1
mkdir control
mkdir exp2
cd ..
cd ..

pwd
                            
cp -r exp1/data/* ./backups/$1/exp1/
cp -r control/data/* ./backups/$1/control/
cp -r exp2/data/* ./backups/$1/exp2/

cd backups
 
tar cf ./backups/$1.tar ./backups/$1/*

fi
