<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!--Gives information that Part III of experiment is finished
    By presing the button, players are redirected to the questionary
-->

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<br><br>

L'expérience est presque finie. Merci pour votre participation.

Avant de pouvoir finaliser l'expérience, vous devez seulement compléter le formulaire suivant.

Après quoi vous verrez combient vous avez gagné durant cette expérience.
 
<?php
$_SESSION['step']="questions1".$_SESSION['language'].".php";
?>

<form name="form1" method="post" action="index.php">
Cliquez <input type="submit" value="ici" class="btn" name="submit"> pour continuer.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->
<?php include_once("html_footer.php"); ?>

