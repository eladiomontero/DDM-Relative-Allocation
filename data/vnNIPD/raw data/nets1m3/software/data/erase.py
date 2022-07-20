#!/usr/bin/python
# This program is used by erase.sh which deletes all the files created by program
# This program is deleting all the files with neighbors, which have the same name as players themselves

import os

def readusers(filename):
  fh=open(filename);
  tmp1=fh.readlines();
  fh.close();
  tmp=[];
  for i in tmp1:
    tmp.append(i.replace('\n',''))
  return tmp;
                
users=readusers('userlist');

for i in users:
  os.remove(i)
  os.remove(i+"history")
  os.remove(i+"lock")
  os.remove(i+"loggedin")
  os.remove(i+"move")
  os.remove(i+"playedby")
  os.remove(i+"ready")
  os.remove(i+"score")
  os.remove(i+"timeend")
  os.remove(i+"timestart")
  os.remove(i+"totalscore")
                     