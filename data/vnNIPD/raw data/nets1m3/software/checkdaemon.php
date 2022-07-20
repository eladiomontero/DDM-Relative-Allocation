<?php
session_start(); // starts PHP session which saves the user's name, IP address of h
                 // and the stage of experiment they are in
                                  // before this line not even comment can be written
?>
                                  

<!--Checks if the daemon is started-->

<?php  

if (trim(file_get_contents('data/deamonstarted'))=="started"){ 
    include_once("login".$_SESSION['language'].".php");
?>

<?php    
}
else {
  include_once("notstarted".$_SESSION['language'].".php");
}
?>
