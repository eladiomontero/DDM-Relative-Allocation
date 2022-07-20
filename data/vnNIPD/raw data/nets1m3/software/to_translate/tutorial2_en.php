<!-- Tutorial page 2 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<b>THE EXPERIMENT </b><br><br>


The experiment consists of an undetermined number of rounds,
and it will take around an hour and never more than two to complete. 
The rules are the same for all participants. 

<br><br>
At the beginning of the experiment you will be paired randomly with an other participant.
You will play with this participant in every round of the experiment. 
Both you and the other participant will be anonimous during the whole experiment. 
Therefore you will never know who is the other person and that person will never know who are you. They will not even know your username.
The amount of money you earn in each round depends of both your choice and the choice made by the other participant.

<br><br>

The gains from every round of the experiment will be summed at the end.
Apart from that money, you also receive an amount of 2.5 euros for participating.

<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial3".$_SESSION['language'].".php";
?>


<form name="form1" method="post" action="index.php">
Click <input type="submit" value="here" class="btn" name="submit"> to continue.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->


<?php include_once("html_footer.php"); ?>