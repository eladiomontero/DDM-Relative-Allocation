<!-- Tutorial page 9 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br>

<b> LES TOURS SUIVANTS <br><br></b>

Dans le prochain tour ainsi que ceux qui suivront, vous verrez un écran comme celui-ci :
<center>
<img src="pics/screen_later.png" height="300" align="center"/>
</center>
<br><br>
Le panneau gauche affiche maintenant les choix effectués par vous et par l'autre joueur durant le tour précédent. <br>
La couleur du gros carré supérieur montre votre choix et le nombre dans ce carré montre combien de points vous avec gagnés au tour précédent.<br>
La couleur de la petite boule inférieure montre le choix de l'autre joueur et le nombre montre ses gains.<br>
<br>

Le panneau droit n'a pas encore changé. Vous pouvez poursuivre pour faire votre prochain choix.
<?php
// changing the stage of experiment into the page which is anauncing the start of the experiment             
// this way after pressing the button, user will be redirected to index page   
// and then to updated stage of experiment
$_SESSION['step']="tutorial6".$_SESSION['language'].".php";
?>

<form name="form1" method="post" action="index.php">
Cliquez <input type="submit" value="ici" class="btn" name="submit"> pour continuer.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->



<?php include_once("html_footer.php"); ?>