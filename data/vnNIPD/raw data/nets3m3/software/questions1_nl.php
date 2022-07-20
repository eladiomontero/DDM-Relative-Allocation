<head>
<link rel="StyleSheet" href="style.css" type="text/css">
<script language="JavaScript" type="text/javascript">
function checkform ( form )
{
  if (form.q_part1.value != "")||(form.q_part2.value == "")||(form.q_part3.value == "")||(form.neighbours.value == "")||(form.heardof.value == "")||(form.namePD.value == "") {
    alert( "Por favor, conteste todas las preguntas." );
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

Beschrijf kort hoe je je keuze hebt gemaakt tussen de twee acties blauw en geel: 
<br>
<TEXTAREA NAME="strategy" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>

Werd je keuze beinvloed door de keuze van de ANDERE SPELER in de vorige ronde?
<br>
 &nbsp;&nbsp;<input type="radio" value="Yes" name="theother"> Ja
 &nbsp;&nbsp;<input type="radio" value="No" name="theother"> Neen
<br>
<br>  
<br>

Werd je keuze beinvloed door wat JIJZELF hebt gedaan in de vorige ronde?
<br>
 &nbsp;&nbsp;<input type="radio" value="Yes" name="you"> Ja
 &nbsp;&nbsp;<input type="radio" value="No" name="you"> Neen
<br>
<br>
<br>

Werd je keuze mee bepaalt door de keuze in de rondes voor de vorige? Indien ja, hoe sterk was de invloed van die informatie op je keuze?
<br>
<TEXTAREA NAME="all_rounds" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>

Komt dit experiment je bekend voor?
<br>
 &nbsp;&nbsp;<input type="radio" value="Yes" name="heardof"> Ja
 &nbsp;&nbsp;<input type="radio" value="No" name="heardof"> Neen
<br>
<br>
<br>

Indien ja, wat is de naam van het experiment en wat weet je over dit experiment?
<br>
<TEXTAREA NAME="namePD" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>

Wat is je geslacht? 
<br>
 &nbsp;&nbsp;<input type="radio" value="male" name="gender"> Man
 &nbsp;&nbsp;<input type="radio" value="female" name="gender"> Vrouw
<br>
<br>
<br>

To welke Universiteit of Hogeschool behoor je? 
<br>
 &nbsp;&nbsp;<input type="radio" value="VUB" name="university"> VUB
 &nbsp;&nbsp;<input type="radio" value="ULB" name="university"> ULB
 &nbsp;&nbsp;<input type="radio" value="Other" name="university"> Andere
<br>
<br>
<br>

Wat is het niveau van je studies? 
<br>
 &nbsp;&nbsp;<input type="radio" value="Bachelor" name="study"> Bachelor
 &nbsp;&nbsp;<input type="radio" value="Master" name="study"> Master
 &nbsp;&nbsp;<input type="radio" value="PhD" name="study"> PhD
 &nbsp;&nbsp;<input type="radio" value="Other" name="study"> Andere
<br>
<br>
<br>
 
Hoe oud ben je?
<br>
<TEXTAREA NAME="Age" COLS=10 ROWS=1></TEXTAREA>
<br>
<br>
<br>
  
Waar had je over dit experiment gehoord? 
<br>
&nbsp;&nbsp;<input type="radio" value="flyer" name="hear"> Flyer
&nbsp;&nbsp;<input type="radio" value="poster" name="hear"> Poster
&nbsp;&nbsp;<input type="radio" value="email" name="hear"> Email
&nbsp;&nbsp;<input type="radio" value="facebook" name="hear"> Facebook
&nbsp;&nbsp;<input type="radio" value="other" name="hear"> Andere <input type="text" name="other_hear" />
<br>
<br>
<br>
     

Als je nog iets kwijt wil over het experiment, laat het ons weten: 
<br>
<TEXTAREA NAME="Comments" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>

<?php
$_SESSION['step']="write.php";
?>


Klik <input type="submit" value="hier" class="btn" name="submit"> om verder te gaan.
</form><br />
<?php include_once("html_footer.php"); ?>

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->
