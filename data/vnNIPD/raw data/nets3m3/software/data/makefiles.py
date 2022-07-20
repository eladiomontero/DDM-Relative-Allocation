#!/usr/bin/python

# Makes all the necessary files

def readusers(filename):
  fh=open(filename);
  tmp1=fh.readlines();
  fh.close();
  tmp=[];
  for i in tmp1:
    tmp.append(i.replace('\n',''))
  return tmp;
  
users=readusers('userlist');

# File which gives the information that it is the first round
# after the first round the content of this file is changed
fh=open("firstround","w");
fh.write("first");
fh.close();

# Makes Log file where python is writing
fh=open("log","w");
fh.write("Round User Move Lock\n");
fh.close();

# Makes Log file there PHP is writing
fh=open("log_php","w");
fh.write("Round User Move Lock\n");
fh.close();

# File with current round
fh=open("round","w");
fh.write("1");
fh.close();

# File with the information if background program is started
# Until the background program is started the players cannot log in
fh=open("experimentstarted","w");
fh.write("not");
fh.close();


# File with the information if background program is started
# Until the background program is started the players cannot log in
fh=open("deamonstarted","w");
fh.write("not");
fh.close();

# File with information if all the players are logged in
# After everybody logs in, the content of this file is changed
# and the players can proceed to read the tutorial 
fh=open("allloggedin","w");
fh.write("not");
fh.close();


# File with information how many players played
fh=open("counterplayed","w"); # probably not necessary
fh.write("0");
fh.close();

# File with information of the total payoff 
fh=open("totaltotal","w");
fh.write("0");
fh.close();  

# File with information how many players read the tutorial is now ready to play
fh=open("counterready","w"); # probably not neccesary
fh.write("0");
fh.close();

# File with information how many players
fh=open("countercalculated","w"); # probably not neccesary
fh.write("0");
fh.close();   

# File with information if the experiment is started
fh=open("started","w");
fh.write("notstarted");
fh.close();


# Files for each of players
for i in users:
  fh=open(i+"score","w");
  fh.write("0\n");
  fh.close();
  fh=open(i+"totalscore","w");
  fh.write("0\n");
  fh.close();     
  fh=open(i+"lock","w");
  fh.write("notplayed");
  fh.close();
  fh=open(i+"history","w");
  fh.write("Round Move Score Time_php[ms] Time_js[ms] PlayedBy\n");  
  fh.close();
  fh=open(i+"move","w");
  fh.close();     
  fh=open(i+"timestart","w");
  fh.write("1.5");
  fh.close();
  fh=open(i+"time","w");
  fh.write("1000.5");
  fh.close();
  fh=open(i+"timeend","w");
  fh.write("2.3");  
  fh.close();            
  fh=open(i+"ready","w");
  fh.write("notready");  
  fh.close();              
  fh=open(i+"playedby","w");
  fh.close();
  fh=open(i+"loggedin","w");
  fh.write("no");
  fh.close();
  fh=open(i+"answers","w");
  fh.write("The Answers\n");
  fh.close();
              
