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

Please describe briefly how you decided to push the blue and yellow buttons:
<br>
<TEXTAREA NAME="strategy" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>

Was your decision influenced by that THE OTHER PLAYER did in the previous round:
<br>
 &nbsp;&nbsp;<input type="radio" value="Yes" name="theother"> Yes
 &nbsp;&nbsp;<input type="radio" value="No" name="theother"> No
<br>
<br>
<br>  

Was your decision influenced by what YOU did in the previous round?
<br>
 &nbsp;&nbsp;<input type="radio" value="Yes" name="you"> Yes
 &nbsp;&nbsp;<input type="radio" value="No" name="you"> No
<br>
<br>
<br>

Was your decision influenced by all the choices that were made in the rounds before the previous one? If yes, to what extent?
<br>
<TEXTAREA NAME="all_rounds" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>



Is this experiment familiar to you? 
<br>
 &nbsp;&nbsp;<input type="radio" value="Yes" name="heardof"> Yes
 &nbsp;&nbsp;<input type="radio" value="No" name="heardof"> No
<br>
<br>
<br>


What is the name of this game? What do you know about it?
<br>
<TEXTAREA NAME="namePD" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>

What is your gender? 
<br>
 &nbsp;&nbsp;<input type="radio" value="male" name="gender"> Male
 &nbsp;&nbsp;<input type="radio" value="female" name="gender"> Female
<br>
<br>
<br>

What is your University or School
<br>
 &nbsp;&nbsp;<input type="radio" value="VUB" name="university"> VUB
 &nbsp;&nbsp;<input type="radio" value="ULB" name="university"> ULB
 &nbsp;&nbsp;<input type="radio" value="Other" name="university"> Other
<br>
<br>
<br>

What is the level of your studies you are currently doing? 
<br>
 &nbsp;&nbsp;<input type="radio" value="Bachelor" name="study"> Bachelor
 &nbsp;&nbsp;<input type="radio" value="Master" name="study"> Master
 &nbsp;&nbsp;<input type="radio" value="PhD" name="study"> PhD
 &nbsp;&nbsp;<input type="radio" value="Other" name="study"> Other 
<br>
<br>
<br>
 
How old are you?
<br>
<TEXTAREA NAME="Age" COLS=10 ROWS=1></TEXTAREA>
<br>
<br>
<br>
  
If you want to make another comment, please use the space below: 
<br>
<TEXTAREA NAME="Comments" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>
<br>

<?php
$_SESSION['step']="write.php";
?>

How did you hear about the experiment? 
<br>
 &nbsp;&nbsp;<input type="radio" value="flyer" name="hear"> Flyer
 &nbsp;&nbsp;<input type="radio" value="poster" name="hear"> Poster
 &nbsp;&nbsp;<input type="radio" value="email" name="hear"> Email
 &nbsp;&nbsp;<input type="radio" value="facebook" name="hear"> Facebook
 &nbsp;&nbsp;<input type="radio" value="other" name="hear"> Other <input type="text" name="other_hear" />
<br>
<br>
<br>


Click <input type="submit" value="here" class="btn" name="submit"> to continue.
</form><br />
<?php include_once("html_footer.php"); ?>

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->
