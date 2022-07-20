<?php
session_start(); // starts PHP session which saves the user's name, IP address of his/hers computer 
                 // and the stage of experiment they are in
                 // before this line not even comment can be written
?>

<!--The index function, which redirects client to the current stage of experiment -->

<html>
<head>

<link rel="shortcut icon" href="favicon.ico" >

<title>BEEL Experiment</title>

</head>
<body>

<?php

if(!isset($_SESSION['compname'])) {
  $hostname = gethostbyaddr($_SERVER['REMOTE_ADDR']);  //saves the IP address of user's computer  
}
?>

<?php

// if the daemon in not started and user is logged in, it automatically logs out the user
if (trim((file_get_contents('data/deamonstarted'))!="started")&&(isset($_SESSION['username']))) include("logout.php"); // logout


// if the user is not logged in and didn't already wrote their username and password then the login page is included
// if the user is not logged in but he/she wrote their username and password the part for checking the data is included
// if the user is logged in then:
//      if the user is SuperUser the monitoring part is included
//      if the user is normal user the part of code for current stage of experiment is included
//      (what is current stage of experiment is saved in session variable called step)
 
if(isset($_SESSION['username'])) 
  if ($_SESSION['username']=="SuperUser") include_once("sumain.php");
  else include_once($_SESSION['step']);
else 
  if(isset($_POST['username'])) include_once("checkuser.php");
  else include_once("first".$_SESSION['language'].".php"); 
  
?>

</div> 
</body>
</html>
