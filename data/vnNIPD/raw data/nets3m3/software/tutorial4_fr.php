<!-- Tutorial page 6 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>
<body>
<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br>
<b> LE PREMIER ROUND </b> <br><br>
Ceci est le premier écran que vous verrez:
<br><br>
<center>
<img src="pics/screen_first_fr.png" height="300" align="middle"/>
</center>
<br><br>

L'écran est divisé en deux panneaux. Dans le panneau plus grand de gauche vous voyez ce qu'il s'est passé dans les précédents rounds et 
dans le panneau de droite vous devez choisir une couleur. 
Dans le premier round, les carrés du panneau gauche sont blancs étant donné qu'il n'y a pas de précédent round. 
<br><br>
Expliquons maintenant plus en détail le panneau de droite.
Dans ce panneau, vous voyez deux boutons: un bleu et un jaune. 
En cliquant sur l'un de ces boutons vous faites votre choix pour ce round de l'expérience. 
<br>
Au milieu du panneau, vous pouvez voir combien de temps il vous reste pour faire votre choix. 
Vous avez 30 secondes pour faire ce choix. 
Rien ne se passera si le temps s'écoule. Cela causera seulement un délai pour tous les participants. 
Ainsi, merci de ne pas retarder sans raison vos réponses pour éviter que l'expérience ne prenne trop longtemps.
<br>
<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial5".$_SESSION['language'].".php";
?>

<form name="form1" method="post" action="index.php">
Cliquez <input type="submit" value="ici" class="btn" name="submit"> pour continuer.
</form><br />
</body>

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->



<?php include_once("html_footer.php"); ?>
