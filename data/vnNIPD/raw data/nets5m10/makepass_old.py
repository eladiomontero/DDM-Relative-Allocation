#!/usr/bin/python


# this program is making the file with passwords
# it takes list of words from the file passwords.php in which in the second column are the passwords
# and creates the file userlist.php in which first column are users names and second column are associated passwords
# usage: ./makepass.py N
# N is maximal total number of players 

from random import *
from sys import *
import random
import sys


N = int(sys.argv[1]) # maximal number of players

ListPass=[]

fh=open("passwords.php");
tmp1=fh.readlines();
fh.close();
                               
for i in range(1,N+1):
  a, b=tmp1[i-1].split("	")
  Element=b
  ListPass.extend([Element])

shuffle(ListPass)
fh=open("./exp1/data/userlist.php","w");

for i in range(1,N+1):
  fh.write('%s %s' % ("usuario"+str(i),ListPass[i-1]));
                              
fh.write("SuperUser suclave\n"); 							# this is the password for superuser who monitors the experiment
                                                                                        #In case one wants to change the password for SuperUser change "suclave"
                                                                                        # into the new password (new password can't have spaces)

fh.write('\n<script type="text/javascript">\ntop.location="index.php";\n</script>\n');  # these line is here in case that somebody opens this file by chance from the web
fh.close()                                                                              # it will redirect it to the index page and they will not be able to see the list of passwords
