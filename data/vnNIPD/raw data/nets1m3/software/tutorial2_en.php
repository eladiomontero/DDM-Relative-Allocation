<!-- Tutorial page 2 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<b>THE EXPERIMENT </b><br><br>


The experiment consists of an undetermined number of rounds. It will take around an hour to complete. 

<b>The rules are the same for all participants. </b>

<br><br>
At the beginning of the experiment all the participants are placed on a random position of a square lattice. Each participant will have 4 neighbours, one directly above, one directly below, one on the left and finally one on the right. 

<br><br>The neighbours are the same during the whole game, meaning that <b>you will play with the same participants in every round of the experiment. </b>

<br><br>
Both you and the other participants will be anonymous during the whole experiment. Your neighbours in the game will most likely be not the people sitting next to you in the room. Therefore you will never know who is the other person and that person will never know who you are. They will not even know your username.

<br><br>
The amount of money you earn in each round depends on both your choice and the choice made by the other participants. The gains from every round of the experiment will be summed at the end.
Apart from that money, you will also receive an amount of 2.5 euros for participating.

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
