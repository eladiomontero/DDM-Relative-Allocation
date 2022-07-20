<!--Presents the screen of the first round
    with the sketch of the player and his neighbors 
    and two buttons on the right side to choose from  
 -->
<?php
  $time_start = microtime(true);               //records the starting time of the moves, so later it could be used to calculate how long it took player to make a move
  $fh=fopen('data/'.$user.'timestart','w+');
  fwrite($fh,$time_start."\n");
  fclose($fh);
//  $rand_button=rand(0, 1);
?>     
<head>
<link rel="StyleSheet" href="stylemain1.css" type="text/css">
</head>
<body>
<script>
  var start = new Date().getTime();
  var count=30;
  var counter=setInterval(timer, 1000); //1000 will  run it every 1 second
  function timer()
  {
    count=count-1;
    if (count>5)
      document.getElementById("timer").innerHTML=count + " secs"; //
    else
      if (count>0)
        {
          document.getElementById("timer").style.color = "red";  // turn red
          document.getElementById("timer").innerHTML=count + " secs!!!"; // 
        }
      else
        document.getElementById("timer").innerHTML=count + " secs. <br> PLEASE, PLAY NOW!!!"; // watch for spelling         
  }      
  function clickme()  
  {  
    var end = new Date().getTime()-start;    
    document.getElementById("timefinalC").value=end;    
    document.getElementById("timefinalD").value=end;    
  }        
</script>

<!-- design of the playing screen in the first round-->
<center>
<table width="100%" height="99.9%" align=center border="0" cellspacing="0" cellpadding="0">
 <col width="25%">
 <col width="25%">
 <col width="25%">
 <col width="25%">

<tr height=10% align=center>
  <td> 
    <table height="100%" width="50%" align=left>
      <col width="40">
      <col width="60">
      <col width="60">
      <tr height="40"> <td> </td> <td bgcolor="0099CC"> </td> <td bgcolor="FFFF33"> </td> </tr>
      <tr height="40"> <td bgcolor="0099CC"> </td> <td align=center> <?php echo trim(file_get_contents('data/reward'))?> </td>
                       <td align=center> <?php echo trim(file_get_contents('data/suckers')) ?> </td>  </tr>
      <tr height="40"> <td bgcolor="FFFF33"> </td> <td align=center> <?php echo trim(file_get_contents('data/temptation'))?> </td>
                       <td align=center> <?php echo trim(file_get_contents('data/punishment')) ?> </td> </tr>
    </table>    
  </td>

  <td align=center>
   <p style="color:777777;font-size:15px"> </p>
  </td>  

  <td align=right>
    <p style="color:777777;font-size:15px"> </p>
  </td>

  <td bgcolor="EEEEEE" valign=top><p style="color:777777"> Time left:</p><br>
   </td> 
       
</tr>  



<tr height="30%">
  <td align=center>
  </td>
  <td align=center>
      <table border="1" width="40%" height="40%"  bgcolor="FFFFFF" >
         <tr><td align=center>Neighbour 1</td></tr></table> </td>         
  <td align=center> </td>
  <td align=center bgcolor="EEEEEE"> 
   <table hight="50%" bgcolor="EEEEEE">
     <tr valign=top>
       <span style="font-size:30px;" id="timer"></span>
     </tr>
     <tr bgcolor="EEEEEE" align=center><br />&nbsp;<br />&nbsp; <p style="color:red; font-size:30px;">Choose a colour:</p> </tr>
   </table>
  </td>
       
</tr>
<tr height="30%">
  <td align=center>
      <table border="1" width="40%" height="40%"  bgcolor="FFFFFF" >
        <tr><td align=center>Neighbour 4</td></tr></table> </td>   
  <td align=center> 
      <table border="10" width="90%" height="90%"  bgcolor="FFFFFF" >
        <tr><td align=center> <p style="font-size:30px;">You</p>  </td></tr></table>  </td>
  <td align=center>
      <table border="1" width="40%" height="40%"  bgcolor="FFFFFF" >
       <tr><td align=center>Neighbour 2</td></tr></table> </td>   
  <td bgcolor="EEEEEE" align=center> <?php include_once("buttonC.php")?> </td>
</tr>
<tr height="30%">
  <td align=center>
  <td align=center>
      <table border="1" width="40%" height="40%"  bgcolor="FFFFFF" >
        <tr><td align=center>Neighbour 3</td></tr></table> </td>             
  <td align=center>
  <td bgcolor="EEEEEE" align=center> <?php include_once("buttonD.php")?> <br />&nbsp;<br />&nbsp; </td> 
</tr>   
        
</table>
</center>


</body>
