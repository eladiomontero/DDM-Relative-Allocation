<!-- Tutorial page 6 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>
<body>
<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br>

<b>DE EERSTE RONDE</b>
<br><br>
Dit is het eerste scherm dat je zal zien:
<br><br>
<center>
<img src="pics/screen_first_nl.png" height="300" align="middle"/>
</center>
<br><br>
Het scherm is opgesplitst in twee panelen.  In het grotere paneel aan de linkerkant zal je kunnen zien wat er gebeurd is in de voorgaande ronde.  In het rechterpaneel zal je je kleur moeten kiezen.  In de eerste ronde zullen de vierkanten in het linkerpaneel wit zijn omdat er geen voorgaande informatie beschikbaar is.
<br><br>
In dit deel van de uitleg wordt het rechterpaneel uitgelegd.  In dat rechterpaneel zijn er twee knoppen voorzien:  een blauwe en een gele.  Door met je muis op een van die knoppen te klikken maak je je keuze in elke ronde van het experiment.
<br>
In het midden van het paneel staat hoeveel tijd er overblijft om je keuze te maken. Je zou je keuze moeten kunnen maken binnen de 30 seconden.  Als de tijd voorbij is gebeurt er niets.  Het resultaat zal echter zijn dat al de andere deelnemers op jou moeten wachten voor de volgende ronde van het experiment kan worden gestart.
Gelieve dus niet te lang te wachten met je keuze omdat anders het experiment te lang zal duren.
<br>
<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial5".$_SESSION['language'].".php";
?>

<form name="form1" method="post" action="index.php">
Druk <input type="submit" value="hier" class="btn" name="submit"> om verder te gaan.
</form><br />
</body>

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->



<?php include_once("html_footer.php"); ?>
