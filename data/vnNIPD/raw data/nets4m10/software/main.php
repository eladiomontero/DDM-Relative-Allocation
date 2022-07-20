<!-- This part reads basic parameters of players:
     - player's username
     - player's neighbors
     - payoffs in the previous round      
    
     Then it forwards the player to one of the following:
     - First playing screen
     - Screen where player should play
     - Screen when player played and now he/she is waiting for everybody else to play
-->

<head>
<link rel="StyleSheet" href="stylemain1.css" type="text/css">
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />

</head>

<!-- Uncomment the fallowing part in the testing phase -->
<!--
<p align=right>
<font size=-1 halign="right">
<a href="logout.php">Logout</a>
</font>
</p>
-->
<script type="text/javascript">    
var TimeStart_js = performance.now();
</script>



<?php

// reads the total number or rounds or reports the error if the roundsnumber file is empty
while (1>0){
  $rn=trim(file_get_contents('data/roundsnumber'));
  if ($rn!='') break;
  $fh=fopen('data/log_php','a+');
  fwrite($fh,$user." had RED in roundsnumber".$rn."\n");
  fclose($fh);                        
}  
                                      
//reads current round number
$round=(int)file_get_contents('data/round');


if($round>=$rn) $_SESSION['step']="end_game".$_SESSION['language'].".php";      

$user=$_SESSION['username'];
$usern=file('data/'.$user);					   // this can be moved to NeighbourN.php files
$score=file('data/'.$user.'score');                                // not necessary here
$score=$score[0];                                                  // not necessary here
$numberofusers=intval(file_get_contents('data/numberofusers'));    // probably not necessary here
  
  for($i=0; $i++; $i<8){					   // this can be moved
   $usern[$i]=substr($usern[$i],0,-1);
  }   
?>


<?php
if(file_get_contents('data/firstround')=='first')
  if(file_get_contents('data/'.$user.'lock')!='notplayed')
    {
    include_once("main_played".$_SESSION['language'].".php"); 
    }
  else 
    { 
      include_once("main_first".$_SESSION['language'].".php");
    } 
else 
  {         
  if(file_get_contents('data/'.$user.'lock')!='notplayed') 
    {
      include_once("main_played".$_SESSION['language'].".php");
    } 
  else 
    {
      foreach($usern as $t){
        $t=substr($t,0,-1);                                        //removing "\n" at the end of the line
      }
      foreach($usern as $t){
        $t=substr($t,0,-1); 
        $showscore=(string)file_get_contents('data/'.$t.'score');  //payoffs in the previous round 
      }
      include_once("main_play".$_SESSION['language'].".php");                               
    }
  }     
?>
</tr>
</table>


