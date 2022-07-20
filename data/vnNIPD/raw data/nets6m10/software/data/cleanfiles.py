#!/usr/bin/python

# Resets the parameters in all the files without starting the experiment.
# PHP programs are reading these files and act according to it's value
# Here we set the parametars so players cannot even enter the experiment.




# function which returns array of user's names 
def readusers(filename):
  fh=open(filename);
  tmp1=fh.readlines();
  fh.close();
  tmp=[];
  for i in tmp1:
    tmp.append(i.replace('\n',''))
  return tmp;
  
users=readusers('userlist');



fh=open("deamonstarted","w");	# background process (daemon) is not started
fh.write("not");
fh.close();

fh=open("started","w");		# all players haven't read the tutorial
fh.write("notstarted");
fh.close();

fh=open("allloggedin","w");	# all players have not logged in
fh.write("not");
fh.close();
 
for i in users:			# about individual players
  fh=open(i+"ready","w");
  fh.write("notready");  
  fh.close();              
  fh=open(i+"loggedin","w");
  fh.write("no");
  fh.close();
        
