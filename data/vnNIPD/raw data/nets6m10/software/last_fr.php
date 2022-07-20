<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<?php
$user=$_SESSION['username'];
$total=file_get_contents('data/'.$user.'totalscore');
$point=file_get_contents('data/pointvalue');
$earned=($total)*$point+2.5;

?>

L'expérience est finie.<br><br>

Vous avez gagné : <b>
<?php echo $earned." &#128;"
?> 
</b> 
<br>
<br>

Les 2,5€ comme payement de présence sont compris. <br><br>

Écrivez ce montant sur le formulaire de participation. <br><br>
 
Veuillez attendre que l'expérimentateur vous appelle.
Lorsque vous serez appelé, apportez votre forumlaire complété avec vos informations bancaires afin d'organiser le virement bancaire.
<br><br>

Vous devrez leurs présenter le document avec votre nom d'utilisateur et mot de passe pour qu'ils puissent vérifier le montant que vous avez gagné.<br><br>

<b>Les instructions doivent être rendues aux expérimentateurs.</b><br><br>

Veuillez ne pas parler des détails de l'expérience à d'autres participants potentiels car cela ruinerait notre étude et nous devrons arrêter les expériences. <br><br>

Merci pour votre participation à cette expérience. <br><br>

Nous espérons que vous avez apprécié l'expérience et que vous participerez encore aux futures expériences que nous organiserons.

<?php include_once("html_footer.php"); ?>

