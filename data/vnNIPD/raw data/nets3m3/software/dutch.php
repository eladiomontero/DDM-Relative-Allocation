<?php
session_start(); // starts PHP session which saves the user's name, IP address of his/hers computer 
                 // and the stage of experiment they are in
                                  // before this line not even comment can be written
$_SESSION['language']='_nl';

include("index.php");
?>
                                  