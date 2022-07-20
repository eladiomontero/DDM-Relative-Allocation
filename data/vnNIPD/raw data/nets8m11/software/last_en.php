<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<?php
$user=$_SESSION['username'];
$total=file_get_contents('data/'.$user.'totalscore');
$point=file_get_contents('data/pointvalue');
$earned=($total)*$point+2.5;

?>

The experiment is finished.<br><br>

You earned: <b>
<?php echo $earned." &#128;"
?> 
</b> 
<br>
<br>

Including the show up fee. <br><br>

Write this amount on the participation form.
 
Please wait until the experimenter calls you.  Once you are called take the completed form with your bank details to the experimenters, to arrange the bank transfer. <br><br>

You will need to show them the paper with username and password so they can verify the amount of money you earned.<br><br>

<b>The instructions must be returned to the experimenters.</b><br><br>

Please, do not talk about the details of the experiment to other potential participants as this will ruin our study and we will be required to stop the experiments. <br><br>

Thank you for participating in the experiment. <br><br>

We hope you enjoyed the experience and that you will participate in some of the future experiments we organize.

<?php include_once("html_footer.php"); ?>

