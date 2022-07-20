<!-- Tutorial page 1 -->


<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>
Cette expérience étudie comment les individus prennent des décisions. <br><br>

Nous ne nous attendons pas à ce que vous adoptez un comportement particulier. <br><br>

Nous ne sommes pas autorisés à vous mentir durant les diverses parties de l'expérience. <br><br>

Vous êtes complètement anonyme durant le jeu et vos actions seront seulement liées à votre nom d'utilisateur qui est uniquement connu de vous-même. <br><br>

Les choix que vous faites durant l'expérience détermineront directement le montant que vous gagnerez. <br><br>


<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page
// and then to the next stage of experiment which is next page of tutorial                                                                       
$_SESSION['step']="tutorial2".$_SESSION['language'].".php"; 
?>

<form name="form1" method="post" action="index.php">
Appuyez <input type="submit" value="ici" class="btn" name="submit"> pour continuer.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->


<?php include_once("html_footer.php"); ?>