<!--Checks if the login and password of player are correct and 
logs in the player if they are or redirect then to try again if they are not -->

<?php
  $lines=file('data/userlist.php');
?>  
  
<?php  
if (trim(file_get_contents('data/deamonstarted'))=="started"){ 
  if(in_array($_POST['username']." ".$_POST['txtpassword']."\n",$lines)) {
    $_SESSION['username']=$_POST['username'];
    $user=$_SESSION['username'];
    if ($user!="SuperUser"){
      $fh=fopen('data/'.$user.'loggedin','w+');
      fwrite($fh,"yes\n");
      fclose($fh);
    }
    $fh=fopen('data/'.$_SESSION['username'].'answers','a+');
    fwrite($fh,$_SESSION['language']."\n");
    fclose($fh);
          
    $_SESSION['step']="waitlogin".$_SESSION['language'].".php";
        
?>
<script type="text/javascript">
top.location="index.php";
</script>
<?php    
  }
  else {
include_once("login2".$_SESSION['language'].".php");

  }
}
else {
  include_once("notstarted".$_SESSION['language'].".php");
}
?>
