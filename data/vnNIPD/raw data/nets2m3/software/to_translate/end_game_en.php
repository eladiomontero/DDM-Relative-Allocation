<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!--Gives information that Part III of experiment is finished
    By presing the button, players are redirected to the questionary
-->

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<br><br>
<b>

The experiment is almost finished. Thank you for your participation.

Before we can end the entire experiment, you only need to complete the following questionnaire.

Afterwards you will see how much you have earned in this experiment.
 
<?php
$_SESSION['step']="questions1".$_SESSION['language'].".php";
?>

<form name="form1" method="post" action="index.php">
Click <input type="submit" value="here" class="btn" name="submit"> to continue.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->
<?php include_once("html_footer.php"); ?>

