
<?php
  session_start();
  // Writes data about the action to the corresponding files and redirects to index
  $user=$_SESSION['username'];
  $time_end = microtime(true);
  $move=$_POST['move']; 
 
  //write to the log file 
  $fh=fopen('data/log_php','a+');
  $round = file_get_contents('data/round');
  fwrite($fh,$user." played ".$move.", round ".$round." at ".$time_end."\n");
  fclose($fh);
  
  //write the time move was made 
  $fh=fopen('data/'.$user.'timeend','w+');
  fwrite($fh,$time_end);
  fclose($fh);


 //write reaction time measured by javascript
  $time=$_POST['time'];
  $fp = fopen('data/'.$user.'time', 'w+');
  fwrite($fp, $time);
  fclose($fp);
         
        
  //writes the move
  $fh=fopen('data/'.$user.'lock','w+');
  fwrite($fh,$move."\n");
  fclose($fh);
  
  //writes that move was made by user
  $fh=fopen('data/'.$user.'playedby','w+');
  fwrite($fh,"user");
  fclose($fh);
      
?>
<script type="text/javascript">
top.location="index.php";
</script>
