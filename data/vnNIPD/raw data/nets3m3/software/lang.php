<?php
session_start(); // starts PHP session which saves the user's name, IP address of h
                 // and the stage of experiment they are in
                 // before this line not even comment can be written
                                  
?>                                  

<!-- Starting page of experiment with welcome message.
     After pressing the button players are redirected to login page -->


<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->

<?php include_once("html_header.php"); ?>

<iframe src="http://free.timeanddate.com/clock/i1ih10sb/n141/tles4/fs16/ftb/th1/ts1" frameborder="0" width="43" height="21" align="right"></iframe>
<center>
<font size=5>
<br><br><br><br>
<form name="form1" method="post" action="english.php">
If you want to do the experiment in English, please click <input type="submit" value="here" class="btn" name="submit"> 
</form><br />

<br><br>

<form name="form1" method="post" action="french.php">
Si vous voulez  faire l'expérience en Français, cliquez <input type="submit" value="ici" class="btn" name="submit"> 
</form><br />

<br><br>

<form name="form1" method="post" action="dutch.php">
Als je het experiment in het nederlands wil doen, druk dan <input type="submit" value="hier" class="btn" name="submit"> 
</form><br />
</center>
</font>
<?php include("html_footer.php")?>