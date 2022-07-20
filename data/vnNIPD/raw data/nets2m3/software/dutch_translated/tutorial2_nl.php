<!-- Tutorial page 2 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<b>HET EXPERIMENT </b><br><br>

The experiment bestaat een aantal rondes die ongeveer een uur in beslag zullen nemen.

<b>De regels zijn hetzelfde voor al de deelnemers. </b>

<br><br>
Aan het begin van het experiment worden al de deelnemers in een random positie in een netwerk geplaatst.  Dit netwerk heeft de vorm van een raamwerk waarbij elke deelnemer vier buren heeft:  Een buur boven, onder, links en rechts.

<br><br>De buren zijn dezelfde gedurende gans het experiment, wat dus betekent dat <b> je elke ronde met dezelfde deelnemers speelt. </b>

<br><br>
Zowel jij als de andere deelnemers zijn anoniem tijdens het experiment.  Je buren in het spel zijn zeer waarschijnlijk niet de personen die naast je in het lokaal zitten.  Je zal daardoor dus nooit weten  wie exact je medespelers zijn en zij zullen niet weten wie jij bent.  Ze zien zelfs niet de gebruikersnaam die je nodig had om in te loggen.

<br><br>
De opbrengst die je verdient in elke ronde van het spel is afhanelijk van de combinatie van jou keuze en die van je vier buren.  De opbrengst van elke ronde wordt opgeteld en vertaald naar een bedrag in euros.  Naast dit bedrag zal je ook 2,5 Euro krijgen voor je deelname.

<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial3".$_SESSION['language'].".php";
?>


<form name="form1" method="post" action="index.php">
Druk <input type="submit" value="here" class="btn" name="submit"> om verder te gaan.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->


<?php include_once("html_footer.php"); ?>
