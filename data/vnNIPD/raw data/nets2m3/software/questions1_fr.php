<head>
<link rel="StyleSheet" href="style.css" type="text/css">
<script language="JavaScript" type="text/javascript">
function checkform ( form )
{
  if (form.q_part1.value != "")||(form.q_part2.value == "")||(form.q_part3.value == "")||(form.neighbours.value == "")||(form.heardof.value == "")||(form.namePD.value == "") {
    alert( "S'il vous plaît répondre à toutes les questions." );
    form.email.focus();
    return false ;
  }
  return true ;
}
</script>
</head>

<!-- Questionary at the end of whole experiment -->

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br>

<form name="form1" method="post" action="index.php">

Veuillez décrire brièvement comment vous décidez d'appuyer sur le bouton bleu et jaune:
<br>
<TEXTAREA NAME="strategy" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>

Est-ce que votre décision a été influencée par ce qu'a fait L'AUTRE JOUEUR lors du tour précédant:
<br>
 &nbsp;&nbsp;<input type="radio" value="Yes" name="theother"> Oui
 &nbsp;&nbsp;<input type="radio" value="No" name="theother"> Non
<br>
<br>
<br>
  
Est-ce que votre décision a été influencée par ce que VOUS avez fait lors du tour précédant:
<br>
 &nbsp;&nbsp;<input type="radio" value="Yes" name="you"> Oui
 &nbsp;&nbsp;<input type="radio" value="No" name="you"> Non
<br>
<br>
<br>

Est-ce que votre décision a été influencée par tous les choix fait durant les tours qui précédaient le tour précédent?
Si oui, en quoi ces choix vous ont-ils influencés ?
<br>
<TEXTAREA NAME="all_rounds" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>


Est-ce que cette expérience est familière pour vous? 
<br>
 &nbsp;&nbsp;<input type="radio" value="Yes" name="heardof"> Oui
 &nbsp;&nbsp;<input type="radio" value="No" name="heardof"> Non
<br>
<br>
<br>

Quel est le nom de ce jeu ? Que connaissez-vous à son propos?
<br>
<TEXTAREA NAME="namePD" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>

Quel est votre genre? 
<br>
 &nbsp;&nbsp;<input type="radio" value="male" name="gender"> Masculin
 &nbsp;&nbsp;<input type="radio" value="female" name="gender"> Féminin
<br>
<br>
<br>

Quel est votre École ou Université?
<br>
 &nbsp;&nbsp;<input type="radio" value="VUB" name="university"> VUB
 &nbsp;&nbsp;<input type="radio" value="ULB" name="university"> ULB
 &nbsp;&nbsp;<input type="radio" value="Other" name="university"> Autre
<br>
<br>
<br>

Quel est le niveau des études que vous êtes en train d'entreprendre? 
<br>
 &nbsp;&nbsp;<input type="radio" value="Bachelor" name="study"> Bachelier
 &nbsp;&nbsp;<input type="radio" value="Master" name="study"> Maîtrise
 &nbsp;&nbsp;<input type="radio" value="PhD" name="study"> Doctorat
 &nbsp;&nbsp;<input type="radio" value="Other" name="study"> Autre
<br>
<br>
<br>
 
Quel âge avez-vous ?
<br>
<TEXTAREA NAME="Age" COLS=10 ROWS=1></TEXTAREA>
<br>
<br>
<br>
  
Ou avez-vous entendu parler de cette expérience? 
<br>
 &nbsp;&nbsp;<input type="radio" value="flyer" name="hear"> Flyer
  &nbsp;&nbsp;<input type="radio" value="poster" name="hear"> Poster
   &nbsp;&nbsp;<input type="radio" value="email" name="hear"> Email
    &nbsp;&nbsp;<input type="radio" value="facebook" name="hear"> Facebook
     &nbsp;&nbsp;<input type="radio" value="other" name="hear"> Autre <input type="text" name="other_hear" />
     <br>
     <br>
     <br>
     

Si vous voulez faire un autre commentaire, veuillez utiliser l'espace ci-dessous :
<br>
<TEXTAREA NAME="Comments" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>

<?php
$_SESSION['step']="write.php";
?>


Cliquez <input type="submit" value="ici" class="btn" name="submit"> pour continuer.
</form><br />
<?php include_once("html_footer.php"); ?>

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->
