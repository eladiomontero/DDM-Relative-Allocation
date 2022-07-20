<!-- Tutorial page 6 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>
<body>
<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br>
This is the first screen you will see:
<br><br>
<center>
<img src="pics/screen_first.png" height="300" align="middle"/>
</center>
<br><br>

Het scherm is verdeeld in twee panelen. In het linker paneel kan je zien wat er gebeurd is in de vorige ronde en in het rechterpaneel zal je een keuze moeten make tussen twee kleuren.
In de eerste ronde zijn de vierkanten in het linkerpaneel wit omdat er dan geen voorgaande informatie beschikbaar is.

<br><br>
In dit deel van de uitleg zal het rechterpaneel worden uitgelegd.
Aan de onderkant van dit rechterpaneel zie je twee knoppen: een blauwe en een gele.
Door op een van deze knoppen te klikken maak je je keuze in elke ronde van het experiment.
<br>
In het midden van dat paneel zal je zien hoeveel resterende tijd je hebt voor het maken van je keuze.  Je zou je keuze moeten kunnen maken in 30 seconden. Er gebeurt niets als de 30 seconden voorbij zijn en je je keuze nog niet gemaakt hebt.  Het zal er enle toe leiden dat de andere deelnemers moten wachten tot jij je keuze hebt gemaakt.
Gelieve binnen de tijd je keuze te maken omdat anders het experiment te lang duurt.

<br>
<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial5".$_SESSION['language'].".php";
?>

<form name="form1" method="post" action="index.php">
Klik <input type="submit" value="here" class="btn" name="submit"> om verder te gaan.
</form><br />
</body>

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->



<?php include_once("html_footer.php"); ?>
