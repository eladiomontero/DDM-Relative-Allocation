#!/usr/bin/python

from sys import *


Np=18

fh=open("./data/userlist.php");
tmp1=fh.readlines();
fh.close();
           
for i in range(1,Np+1):
  a, b=tmp1[i-1].split(" ")
  user=a
  password=b.replace("\n","")
  Url='u'+str(i)
  fh=open(Url+".php","w") 
  fh.write("<?php\n$korisnik="+str(user)+";\n$sifra="+str(password)+";\ninclude_once(\"login.php\");\n?>\n")
  fh.close()
                      
                        

    












