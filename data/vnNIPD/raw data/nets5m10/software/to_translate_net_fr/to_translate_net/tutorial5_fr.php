<!-- Tutorial page 9 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br>

<b> LES ROUNDS SUIVANTS<br><br></b>

Dans le second round ainsi que les suivants, vous verrez l'écran suivant:
<center>
<img src="pics/screen_later.png" height="300" align="center"/>
</center>
<br><br>

Le panneau de gauche montre maintenant les choix et gains pour vous et vos voisins lors du précédent round.<br>
La couleur du grand carré au milieu montre votre choix, et le nombre à l'intérieur correspond au montant que vous avez gagné au précédent round. <br>
La couleur des plus petits carrés montre les choix de vos voisins, et les nombres leurs gains.<br>
<br>

Le panneau de droite n'a pas changé. Vous pouvez y faire votre choix suivant.
<?php
// changing the stage of experiment into the page which is anauncing the start of the experiment             
// this way after pressing the button, user will be redirected to index page   
// and then to updated stage of experiment
$_SESSION['step']="tutorial6".$_SESSION['language'].".php";
?>

<form name="form1" method="post" action="index.php">
Cliquez <input type="submit" value="here" class="btn" name="submit"> pour continuer.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->



<?php include_once("html_footer.php"); ?>
