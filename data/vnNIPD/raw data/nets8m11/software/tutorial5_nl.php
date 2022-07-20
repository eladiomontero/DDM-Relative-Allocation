<!-- Tutorial page 9 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br>

<b> DE VOLGENDE RONDES <br><br></b>

In de tweede en al de daarop volgende rondes, zal je een scherm zoals dit zien:
<center>
<img src="pics/screen_later_nl.png" height="300" align="center"/>
</center>
<br><br>

Het linkerpaneel laat je nu de keuzes en de punten die werden vergaard in de vorige ronde door jou en je vier buren zien. <br>
De kleur van het middelste grote vierkant komt overeen met jou keuze en het getal in dat vierkant toont de punten die je verdiende in de vorige ronde.<br>
De kleuren in de kleinere vierkanten laten de keuzes van elke buur zien als ook de punten die zij verdienden met hun keuze.<br>
<br>
Het rechterpaneel is niet veranderd.  Je kan gewoon je volgende keuze maken.

<?php
// changing the stage of experiment into the page which is anauncing the start of the experiment             
// this way after pressing the button, user will be redirected to index page   
// and then to updated stage of experiment
$_SESSION['step']="tutorial6".$_SESSION['language'].".php";
?>

<form name="form1" method="post" action="index.php">
Druk <input type="submit" value="hier" class="btn" name="submit"> om verder te gaan.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->



<?php include_once("html_footer.php"); ?>
