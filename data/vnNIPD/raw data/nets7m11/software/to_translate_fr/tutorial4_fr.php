<!-- Tutorial page 6 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>
<body>
<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br>
Ceci est le premier écran que vous verrez :
<br><br>
<center>
<img src="pics/screen_first.png" height="300" align="middle"/>
</center>
<br><br>

Cet écran est divisé en deux panneaux. Dans le panneau de gauche, vous pouvez voir ce qui s'est passé durant le tour précédent et dans le panneau de droite,
vous devez choisir une couleur.
Dans le premier tour, les carrés dans le panneau de gauche sont blancs parce qu'il n'y a pas encore d'information sur le tour précédent.
<br><br>
Dans cette partie du tutoriel, j'explique plus en détail le panneau gauche.
Le bas du panneau droit présente deux boutons : un bleu et un jaune.
À chaque tour de cette expérience, vous effectuerez votre choix en cliquant sur l'un de ces boutons.
<br>
Au milieu du panneau, vous pouvez voir combien de temps il reste pour faire votre choix.
Vous devriez être en mesure de faire votre choix dans les 30 secondes.
Rien ne se passe lorsque le temps est épuisé. Cela resultera seulement en un retard pour tous les autres participants.
Veuillez prêter attention au temps, sinon l'expérience pourrait durer trop longtemps.

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
