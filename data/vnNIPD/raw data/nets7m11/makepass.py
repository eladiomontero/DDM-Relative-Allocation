#!/usr/bin/python


# This program is making the file with passwords called userlist.php
# it takes list of words from the file passwords.php in which are the passwords
# and creates the file userlist.php in which first column are users names and 
# the second column are passwords which are randomly associated to usernames 
# usage: ./makepass.py N
# N is maximal total number of players 

from random import *
from sys import *
import random
import sys


N = int(sys.argv[1]) # maximal number of players

ListPass=[]

fh=open("passwords.php");	     # reads the password list
tmp1=fh.readlines();
fh.close();
                               
for i in range(1,N+1):				
  ListPass.extend([tmp1[i]])

shuffle(ListPass)		    # shuffels the password list so it can be randomly associated to the usernames

fh=open("./software/data/userlist.php","w");
fh1=open("./software/data/userlist_print.dat","w");  

for i in range(1,N+1):
  fh.write('%s %s' % ("usuario"+str(i),ListPass[i-1]));			# writes: "usuarioN password"
                                                                        # where usuarioN is username of the player with N being ordinal number of player
                                                                        # and password being one of the words from the list

  fh1.write('username: %s\npassword: %s \n\n' % ("usuario"+str(i),ListPass[i-1]));
fh1.close()                                 
fh.write("SuperUser suclave\n"); 							# this is the password for superuser who monitors the experiment
                                                                                        #In case one wants to change the password for SuperUser change "suclave"
                                                                                        # into the new password (new password can't have spaces)

fh.write('\n<script type="text/javascript">\ntop.location="index.php";\n</script>\n');  # these line is here in case that somebody opens this file by chance from the web
fh.close()                                                                              # it will redirect it to the index page and they will not be able to see the list of passwords
