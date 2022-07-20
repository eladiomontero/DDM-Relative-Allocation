<?php
// After the player reads the tutorial, he/she is redirected to this part
// which changes the content of users file into ready
// so the background program will know that specific player is ready
// When all the players are ready the background program will let them continue with the game

  $user=$_SESSION['username'];  
  file_put_contents('data/'.$user.'ready', 'ready');
  $_SESSION['step']="wait".$_SESSION['language'].".php"; //redirects to wait
?>  
            

<script type="text/javascript">
top.location="index.php";
</script>
                   