<!-- Tutorial page 1 -->


<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>
This experiment studies how individuals make decisions. <br><br>

We do not expect any particular behavior from you. <br><br>

We are not allowed to lie to you in any part of the experiment. <br><br>

You are completely anonymous during the game and your actions will be linked only to your username which is known only to you. <br><br>

The choices you make in the experiment will directly determine the amount of money you earn. <br><br>


<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page
// and then to the next stage of experiment which is next page of tutorial                                                                       
$_SESSION['step']="tutorial2".$_SESSION['language'].".php"; 
?>

<form name="form1" method="post" action="index.php">
Press <input type="submit" value="here" class="btn" name="submit"> to continue.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->


<?php include_once("html_footer.php"); ?>