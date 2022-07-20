<!-- Tutorial page 9 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br>

<b> DE VOLGENDE RONDES<br><br></b>

In zowel de tweede als de volgende rondes zal je een scherm zien zoals:
<center>
<img src="pics/screen_later.png" height="300" align="center"/>
</center>
<br><br>

Het linkerpaneel zal nu de keuze die zowel jij als de andere deelnemer hebben gemaakt in de vorige ronde. <br>

De kleur van het bovenste grote vierkant geeft jou keuze weer en het getal in dat vierkant geeft aan hoeveel punten je hebt gekregen in die ronde.<br>

De kleur van het onderste vierkant geeft de keuze van de andere deelnemer aan en het getal geeft aan heel de andere deelnemer heeft gekregen in die ronde.

<br>
<br>

Het rechterpaneel is niet veranderd.  Je moet nu gewoon opnieuw een keuze maken.
<?php
// changing the stage of experiment into the page which is anauncing the start of the experiment             
// this way after pressing the button, user will be redirected to index page   
// and then to updated stage of experiment
$_SESSION['step']="tutorial6".$_SESSION['language'].".php";
?>

<form name="form1" method="post" action="index.php">
Klik <input type="submit" value="here" class="btn" name="submit"> om verder te gaan.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->



<?php include_once("html_footer.php"); ?>