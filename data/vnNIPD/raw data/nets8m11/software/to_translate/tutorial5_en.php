<!-- Tutorial page 9 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br>

<b> THE FOLLOWING ROUNDS <br><br></b>

In the second and allow following rounds, you will see a screen like this:
<center>
<img src="pics/screen_later.png" height="300" align="center"/>
</center>
<br><br>

The left panel will now show the choices that both you and the other player made in the previous round. <br>
The color of the top big square shows your choice and the number in that square shows how many points you earned in the previous round.<br>
The color of the bottom small square shows the choice of the other player and the number shows their earnings.<br>
<br>

The right panel has not changed. You can proceed to make your next choice.
<?php
// changing the stage of experiment into the page which is anauncing the start of the experiment             
// this way after pressing the button, user will be redirected to index page   
// and then to updated stage of experiment
$_SESSION['step']="tutorial6".$_SESSION['language'].".php";
?>

<form name="form1" method="post" action="index.php">
Click <input type="submit" value="here" class="btn" name="submit"> to continue.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->



<?php include_once("html_footer.php"); ?>