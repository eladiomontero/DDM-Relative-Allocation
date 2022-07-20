#!/usr/bin/python

from random import *
from sys import *
import random
import sys

# reading parameters: network size NxM
# number of rounds between Rmin and Rmax
N = int(sys.argv[1]) # total number of participants
Rmin = int(sys.argv[2])
Rmax = int(sys.argv[3])


if N % 2==1:
  print "Total number of paticipants need to be even"
else:
  ListPrisoners=[]
  rn=random.randint(Rmin, Rmax) # randomly generate number of rounds
  print "Number of rounds in part 1: "+str(rn)
  s = str(rn)
  fh=open("roundsnumber","w")
  fh.write(s)
  fh.close()

  fh=open("numberofusers","w")  # writing total number of players for later use
  fh.write('%d\n' % (N))   
  fh.close()

  fh=open("N","w")              #reads the size of the networks as NxM
  fh.write('%d\n' % (N))   
  fh.close()  


  # reading players' usernames and putting them into the array ListPrisoners 
  fh=open("userlist.php");
  tmp1=fh.readlines();
  fh.close();                               
  for i in range(0,N):
    a, b=tmp1[i].split(" ")
    Element=a
    ListPrisoners.extend([Element])

  # Shuffling the list so that neighbors are random
  shuffle(ListPrisoners)

  # writing the list of players by order in the game
  fh=open("userlist","w");
  for i in range(0,N):
    fh.write('%s\n' % (ListPrisoners[i]));
  fh.close()

# writing the neighbors of each player according to their position in the network 
  for i in range(0,N):
    fh=open(ListPrisoners[i],"w+");
    if i % 2==1: 
      fh.write("%s\n" % (ListPrisoners[i-1]))
    else:
      fh.write("%s\n" % (ListPrisoners[i+1]))
    fh.close()

        