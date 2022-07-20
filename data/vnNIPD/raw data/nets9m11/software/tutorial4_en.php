<!-- Tutorial page 6 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>
<body>
<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br>
<b> THE FIRST SCREEN </b> <br><br>

This is the first screen you will see:
<br><br>
<center>
<img src="pics/screen_first_en.png" height="300" align="middle"/>
</center>
<br><br>

The screen is divided in two panels. In the bigger left panel you can see what happend in the previous round and in the right panel you will have to choose a color. 
In the first round the squares in the left panel are white because there was no previous information.
<br><br>
In this part of the tutorial I will explain in more detail the right panel.
In the right panel you see two buttons: a blue and a yellow button. 
By clicking one of these buttons you make your choice in that round of the experiment.
<br>
In the middle of the panel you can see how much time
is left for you to make your choice.
You should be able to make your choice within the 30 seconds.
Nothing happens when the time runs out.  It only will result in a delay for all the other participants.
Please, do not delay your choice without a reason, otherwise the experiment might take too long.
<br>
<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial5".$_SESSION['language'].".php";
?>

<form name="form1" method="post" action="index.php">
Click <input type="submit" value="here" class="btn" name="submit"> to continue.
</form><br />
</body>

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->



<?php include_once("html_footer.php"); ?>
