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
        document.getElementById("timer").innerHTML=count + " secs. <br> VEUILLEZ JOUER MAINTENANT !!!"; // watch for spelling         
  }      
</script>

<center>

<table width="100%" height="100%" align=center border="0" cellspacing="0" cellpadding="0">
  <col width="50%">
  <col width="30%">
  <col width="20%"
<tr height=10 align=center>
  <td align=center>
   <p style="color:777777;font-size:15px"> TOUR PRÉCÉDENT </p>
  </td>  
  <td bgcolor="EEEEEE" align=right>
    <p style="color:777777;font-size:15px"> TOUR COURANT </p>
  </td>
  <td bgcolor="EEEEEE"> 
    <table height="30%" width="50%" align=right>
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
</tr>  

<tr height="45%" align=center>
  <td align=center valign=center> 
      <table border="10" width="60%" height="80%"  bgcolor="FFFFFF" >
        <tr><td align=center> <p style="font-size:30px;">Vous</p>  </td></tr>
      </table>  
  </td>
  
  <td bgcolor="EEEEEE" colspan=2>
   <table hight="80%">
     <tr valign=top><p style="color:777777"> Temps restant :</p><br>
     </tr> 
     <tr>
       <span style="font-size:30px;" id="timer"></span>
     </tr>
   </table>
  </td>
</tr>

<tr height="45%" align=center>

  <td align=center>
      <table border="1" width="30%" height="40%"  bgcolor="FFFFFF" >
         <tr><td align=center>L'autre participant</td></tr>
      </table> 
  </td>
 
  <td bgcolor="EEEEEE" colspan=2>
   <table align=center valign=center height=80% width=60%>
     <tr rowspan=2 bgcolor="EEEEEE" align=center> <p style="color:red;font-size:30px">Choisissez une couleur :</p> 
     </tr>
    
     <tr>
      <td bgcolor="EEEEEE" align=left> <?php // if($rand_button==0) 
                                          include_once("buttonC.php")?> 
      </td>

      <td bgcolor="EEEEEE" align=right> <?php //if($rand_button==1) 
                                           include_once("buttonD.php")?>
      </td>
     </tr> 
   
   </table>  
  </td> 
</tr>
</table>
</center>
</body>