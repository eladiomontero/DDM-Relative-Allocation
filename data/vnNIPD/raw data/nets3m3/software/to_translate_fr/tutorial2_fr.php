<!-- Tutorial page 2 -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<b>L'EXPÉRIENCE </b><br><br>


L'expérience consiste en un nombre non déterminé de tours.
Elle se déroulera durant environ une heure et jamais plus de deux.
Les règles sont les mêmes pour tous les participants.

<br><br>
Au début de l'expérience, vous serez assigné à un autre joueur sélectionné aléatoirement.
Vous jouerez avec ce participant à chaque tour de l'expérience.
Chacun de vous resterez anonyme durant toute l'expérience.
Par conséquent, vous ne saurez jamais qui est l'autre personne et cette personne ne saura jamais qui vous êtes, même pas votre nom d'utilisateur.
Le montant gagné à chaque tour dépend à la fois de votre choix et de celui de l'autre participant.

<br><br>

Les gains issus de chacun des tours de l'expérience seront additionnés à la fin.
Mis à part cet argent, vous recevez également un montant de 2,5 euros pour la participation.

<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial3".$_SESSION['language'].".php";
?>


<form name="form1" method="post" action="index.php">
Cliquez <input type="submit" value="ici" class="btn" name="submit"> pour continuer.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->


<?php include_once("html_footer.php"); ?>