<!--Announcement of the start of the first part of experiment 
After pressing the button, players are redirected to the tutorial-->


<?php
//session_start();
?>


<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>


<b> L'EXPÉRIENCE  <br><br></b>


Les intructions sont finies. <br><br>
Au prochain écran, l'expérience commencera. <br><br>


<?php
// changing the stage of experiment into the ready
  $user=$_SESSION['username'];
  $_SESSION['step']="ready.php";
?>        
        
<form name="form1" method="post" action="index.php">
Cliquez <input type="submit" value="ici" class="btn" name="submit"> pour commencer l'expérience.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->


<?php include_once("html_footer.php"); ?>
