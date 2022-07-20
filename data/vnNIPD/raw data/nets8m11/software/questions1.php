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
<br><br><br><br><br><br><br><br>

<form name="form1" method="post" action="index.php">

Please describe briefly how did you make your choices:
<TEXTAREA NAME="strategy" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>

Was your choice influenced by that the other player did in the previous round:
<TEXTAREA NAME="previous_round" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>

Was your desion influenced by what the other player did in the all previous rounds:
<TEXTAREA NAME="all_rounds" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>

Is this experiment familiar to you? <br>
 &nbsp;&nbsp;<input type="radio" value="Yes" name="heardof"> Yes
 &nbsp;&nbsp;<input type="radio" value="No" name="heardof"> No
<br>
<br>


What is the name of this game? What do you know about it?
<TEXTAREA NAME="namePD" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>

What is you gender? <br>
 &nbsp;&nbsp;<input type="radio" value="male" name="gender"> Male
 &nbsp;&nbsp;<input type="radio" value="female" name="gender"> Female
<br>
<br>

What is your University <br>
 &nbsp;&nbsp;<input type="radio" value="VUB" name="university"> VUB
 &nbsp;&nbsp;<input type="radio" value="ULB" name="university"> ULB
<br>
<br>

What is the level of your studies? I am currently doing my <br>
 &nbsp;&nbsp;<input type="radio" value="Bachelor" name="study"> Bachelor
 &nbsp;&nbsp;<input type="radio" value="Master" name="study"> Master
 &nbsp;&nbsp;<input type="radio" value="PhD" name="study"> Phd
 &nbsp;&nbsp;<input type="radio" value="Other" name="study"> Other
<br>
<br>
 
How old are you?
<TEXTAREA NAME="Age" COLS=10 ROWS=1></TEXTAREA>
<br>
<br>
  
If you want to make another comment, please use the space below: 
<TEXTAREA NAME="Comments" COLS=100 ROWS=3></TEXTAREA>
<br>
<br>

<?php
$_SESSION['step']="write.php";
?>


Click <input type="submit" value="here" class="btn" name="submit"> to continue.
</form><br />
<?php include_once("html_footer.php"); ?>

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->
