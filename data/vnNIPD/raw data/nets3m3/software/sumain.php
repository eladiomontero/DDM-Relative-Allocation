<!-- This is the main function of monitor
     It presents the current stage of every player, their actions and payoffs
-->

<head>
<link rel="StyleSheet" href="stylemain1.css" type="text/css">
<title>BEEL</title>
</head>

<html>
<?php 

$round=(int)file_get_contents('data/round');
while (1>0){                                                    //check if the rounds number is properly read, if not reads again
  $roundsnumber=trim(file_get_contents('data/roundsnumber'));
  if ($roundsnumber!='') break;
  $fh=fopen('data/log_php','a+');
  fwrite($fh,$user." had RED in roundsnumber".$roundsnumber."\n");
  fclose($fh);
}
          
$N=file_get_contents('data/N'); 
$users=file('data/userlist');
$totaltotal=file_get_contents('data/totaltotal');
$white='FFFFFF';
$black='000000';
$green='00FF33';
$blue='0099CC';
$yellow='FFFF33';
$red='FF0000';
$gray='888888';
$users=file('data/userlist');
?>

<table height="100%" width="100%">
<tr>
<td>
<!-- The Legend of the meanings of the colors--> 
<span style="font-size:12px; background-color:888888;">Not logged in</span> &nbsp;&nbsp; 
<br>
<span style="font-size:12px; background-color:00FF33;">Logged in, but not read the tutorial</span>&nbsp;&nbsp;
<br>
<span style="font-size:12px; background-color:FFFFFF;">Read the tutorial, not played</span>&nbsp;&nbsp;
<br>
<span style="font-size:12px; background-color:FFFF33;">Played yellow</span>&nbsp;&nbsp;
<br>
<span style="font-size:12px; background-color:0099CC;">Played blue</span>&nbsp;&nbsp;

<!-- Presents current round and total earnings for all users in current part-->
<p style="font-size:20px;color:red"> <?php echo 'round '.$round.' of '.$roundsnumber.' <br> Total number of points '.$totaltotal.'</br>' ?> </p>

<a href="plus5rounds.php">Plus 5 rounds</a>  <br> <!-- Button to prolong experiment (warning not working properly)-->
<a href="stop.php">Stop now!</a> <br>             <!-- Button to stop the experiment -->

</td>
<td>
<!--Table with information about all players -->
<center>
<table width="100%" height="100%" align=center border="2" cellspacing="2" cellpadding="2" bgcolor="FFFFFF">

<?php
for ($i=0; $i<($N/2); $i++){
?>
<tr>
<?php  
  for ($j=0; $j<2; $j++){

    $name = $users[$j+$i*2];
    $name=substr($name,0,-1);   
//    mark:
    while (1>0){
      $lock=substr(file_get_contents('data/'.$name.'lock'),0,1);  
      if ($lock!='') break;
      $fh=fopen('data/log_php','a+');
      fwrite($fh,$user." had RED in lock".$lock."end in round ".$round." at ".$time_end."\n");
      fclose($fh); // change this!!!                            
    }  
    while (1>0){
      $ready=file_get_contents('data/'.$name.'ready');
      if ($ready!='') break;
      $fh=fopen('data/log_php','a+');
      fwrite($fh,$user." had RED in ready".$lock."end in round ".$round." at ".$time_end."\n");
      fclose($fh); // change this!!!
    }                                     
    while (1>0){
      $loggedin=file_get_contents('data/'.$name.'loggedin');
      if ($loggedin!='') break;
      $fh=fopen('data/log_php','a+');
      fwrite($fh,$user." had RED in ready".$lock."end in round ".$round." at ".$time_end."\n");
      fclose($fh); // change this!!!
    }         
    if($loggedin=='no') $color=$gray;
    else if($ready!='ready') $color=$green;
        else if($lock=='n') $color=$white;
        else if($lock=='C') $color=$blue;
        else if($lock=='D') $color=$yellow;
        else { 
          $color=$red;
          $fh=fopen('data/log_php','a+');
          fwrite($fh,$user." had RED after in lock:".$lock."end in round ".$round." at ".$time_end."\n");
          fwrite($fh,$user." had RED after in ready:".$ready."end in round ".$round." at ".$time_end."\n"); 
          fclose($fh); // change this!!!
//        goto mark; 
        }                
    echo '<td align=center width=10 height=10  bgcolor="'.$color.'"> <p style="font-size:10px;">'; 
    $tscore = file_get_contents('data/'.$name.'score');
    $ttotal = file_get_contents('data/'.$name.'totalscore');
    echo $name.'<br>';
    echo 'score '.$tscore.'<br>';
    echo 'total '.$ttotal;
?> </td>

<?php
  }
?>  
</tr>  
<?php
}
?> 

</table>
</center>

</td>
</tr>
</table>
<script type="text/javascript">
setTimeout('top.location="index.php"',5000);
</script>
</html>