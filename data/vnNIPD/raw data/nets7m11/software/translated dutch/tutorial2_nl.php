<!-- Tutorial page 2 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<b>HET EXPERIMENT </b><br><br>

Het experiment bestaat uit een onbepaald aantal rondes en zal ongeveer een, en nooit meer dan twee uur duren.  De regels van het experiment zijn hetzelfde voor alle deelnemers.

<br><br>
Aan het begin van het experiment zal op een random manier een andere deelnemer aan jou worden toegewezen. Je speelt met deze andere deelnemer in elk van de rondes van het expriment.

<br><br>
Zowel jij als de andere deelnemer zijn anoniem in gasn het experiment.  Jij zal dus nooit weten wie de andere speler is en die andere speler zal nooit weten wij jij bent.  Ze kennen zelfs je gebruikersnaam niet.  De hoeveelheid centen die je verdient in elke ronde van het experiment is afhankelijk van zowel jou keuze als die van de andere deelnemer.

<br><br>

De opbrengst van elke ronde zal worden samengetelt op het einde van het experiment.
Naast die opbrengst zal he ook 2.5 euro krijgen voor je deelname.

<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial3".$_SESSION['language'].".php";
?>


<form name="form1" method="post" action="index.php">
Klik <input type="submit" value="here" class="btn" name="submit"> om verder te gaan.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->


<?php include_once("html_footer.php"); ?>