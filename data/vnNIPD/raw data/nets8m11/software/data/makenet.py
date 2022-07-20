#!/usr/bin/python

from random import *
from sys import *
import random
import sys

# reading parameters: network size NxM
# number of rounds between Rmin and Rmax
N = int(sys.argv[1]) # change size of network
M = int(sys.argv[2])
Rmin = int(sys.argv[3])
Rmax = int(sys.argv[4])

Np=N*M
ListPrisoners=[]

rn=random.randint(Rmin, Rmax) # randomly generate number of rounds
print "Number of rounds in part 1: "+str(rn)
s = str(rn)
fh=open("roundsnumber","w")
fh.write(s)
fh.close()

fh=open("numberofusers","w")  # writing total number of players for later use
fh.write('%d\n' % (Np))   
fh.close()

fh=open("N","w")              #reads the size of the networks as NxM
fh.write('%d\n' % (N))   
fh.close()  

fh=open("M","w")
fh.write('%d\n' % (M))   
fh.close()  

# reading players' usernames and putting them into the array ListPrisoners 
fh=open("userlist.php");
tmp1=fh.readlines();
fh.close();                               
for i in range(1,Np+1):
  a, b=tmp1[i-1].split(" ")
  Element=a
  ListPrisoners.extend([Element])

# Shuffling the list so that neighbors are random
shuffle(ListPrisoners)

# writing the list of players by order in the network 
fh=open("userlist","w");
for i in range(1,N+1):
  for j in range(1,M+1):
    fh.write('%s\n' % (ListPrisoners[(i-1)*M+j-1]));
fh.close()


# writing the positions of players in the network                              
fh=open("PrisonersNet","w");    
for i in range(1,N+1):
  for j in range(1,M+1):
    fh.write('%s %d %d\n' % (ListPrisoners[(i-1)*M+j-1],i,j))
fh.close()


# writing the neighbors of each player according to their position in the network 
for i in range(1,N+1):
  for j in range(1,M+1):
    fh=open(ListPrisoners[(i-1)*M+j-1],"w+");
#    if i>1 and j>1: 
#      fh.write("%s\n" % (ListPrisoners[((i-1)-1)*M+(j-1-1)]))
#    elif i>1 and j==1:
#      fh.write("%s\n" % (ListPrisoners[((i-1)-1)*M+M-1]))
#    elif i==1 and j>1:
#      fh.write("%s\n" % (ListPrisoners[(N-1)*M+(j-1-1)]))
#    else: 
#      fh.write("%s\n" % (ListPrisoners[(N-1)*M+M-1]))
               
    if i>1: 
      fh.write("%s\n" % (ListPrisoners[((i-1)-1)*M+j-1]))
    else:
      fh.write("%s\n" % (ListPrisoners[(N-1)*M+j-1]))
    
    
#    if i>1 and j<M: 
#      fh.write("%s\n" % (ListPrisoners[((i-1)-1)*M+(j+1-1)]))
#    elif i>1 and j==M:
#      fh.write("%s\n" % (ListPrisoners[((i-1)-1)*M+1-1]))
#    elif i==1 and j<M:
#      fh.write("%s\n" % (ListPrisoners[(N-1)*M+(j+1-1)]))
#    else: 
#      fh.write("%s\n" % (ListPrisoners[(N-1)*M+1-1]))
                                           
    if j<M:         
      fh.write("%s\n" % (ListPrisoners[(i-1)*M+j+1-1]))  
    else:
      fh.write("%s\n" % (ListPrisoners[(i-1)*M+1-1]))  

#    if i<N and j<M:
#      fh.write("%s\n" % (ListPrisoners[((i+1)-1)*M+(j+1-1)]))
#    elif i<N and j==M:
#      fh.write("%s\n" % (ListPrisoners[((i+1)-1)*M+1-1]))
#    elif i==N and j<M:
#      fh.write("%s\n" % (ListPrisoners[(1-1)*M+(j+1-1)]))
#    else:
#      fh.write("%s\n" % (ListPrisoners[(1-1)*M+1-1]))

    if i<N:
      fh.write("%s\n" % (ListPrisoners[((i+1)-1)*M+j-1]))
    else:
      fh.write("%s\n" % (ListPrisoners[(1-1)*M+j-1]))
                                                        
#    if i<N and j>1:
#      fh.write("%s\n" % (ListPrisoners[((i+1)-1)*M+(j-1-1)]))
#    elif i<N and j==1:
#      fh.write("%s\n" % (ListPrisoners[((i+1)-1)*M+M-1]))
#    elif i==N and j>1:
#      fh.write("%s\n" % (ListPrisoners[(1-1)*M+(j-1-1)]))
#    else:
#      fh.write("%s\n" % (ListPrisoners[(1-1)*M+M-1]))

    if j>1:
      fh.write("%s\n" % (ListPrisoners[(i-1)*M+j-1-1]))  
    else:      
      fh.write("%s\n" % (ListPrisoners[(i-1)*M+M-1]))                                                                                       
 
    fh.close()

        