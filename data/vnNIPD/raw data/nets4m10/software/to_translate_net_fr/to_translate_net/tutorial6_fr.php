<!-- Tutorial page 8 - the test -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br>

<b>TESTEZ VOTRE COMPREHENSION</b>
<br>

Merci de répondre à ces questions pour testez votre compréhension de l'expérience. <br>
Combien gagneriez-VOUS dans les situations suivantes? La table avec les gains se trouve sur la droite si besoin est.

<br>
<form name="form1" method="post" action="index.php">

<table width=100%>
<tr>
<td>

<table border="0" cellspacing="10" cellpading align="center">
<col width="50%"> 
<col width="0%"> 
<col width="50%">
  
<tr>
  <td  align="center">
<img src="pics/question1.png" height="180"> <br>
<?php echo trim(file_get_contents('data/suckers'))?> <input type="radio" value="Correct" name="question1"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<?php echo trim(file_get_contents('data/temptation'))?> <input type="radio" value="Incorrect1" name="question1"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<?php echo trim(file_get_contents('data/punishment'))?> <input type="radio" value="Incorrect2" name="question1"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<?php echo trim(file_get_contents('data/reward'))?> <input type="radio" value="Incorrect3" name="question1"> 
<br>
<br>

</td>
<td></td> 
<td align="center">
<img src="pics/question2.png" height="180"> <br>  
<?php echo trim(file_get_contents('data/reward'))?><input type="radio" value="Incorrect1" name="question2"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp 
<?php echo trim(file_get_contents('data/suckers'))?><input type="radio" value="Incorrect2" name="question2"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
<?php echo trim(file_get_contents('data/punishment'))?><input type="radio" value="Correct" name="question2"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
<?php echo trim(file_get_contents('data/temptation'))?><input type="radio" value="Incorrect3" name="question2"> 
<br>
<br>

</td>
</tr>
<tr>
 <td align="center">                 
<img src="pics/question3.png" height="180"> <br>
<?php echo trim(file_get_contents('data/suckers'))?><input type="radio" value="Incorrect1" name="question3"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<?php echo trim(file_get_contents('data/temptation'))?><input type="radio" value="Correct" name="question3"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
<?php echo trim(file_get_contents('data/reward'))?><input type="radio" value="Incorect2" name="question3"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<?php echo trim(file_get_contents('data/punishment'))?><input type="radio" value="Incorrect3" name="question3"> 
<br>
<br>
 </td>
<td></td>  
 <td align="center">
<img src="pics/question4.png" height="180"> <br>
<?php echo trim(file_get_contents('data/punishment'))?><input type="radio" value="Incorrect1" name="question4"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<?php echo trim(file_get_contents('data/suckers'))?><input type="radio" value="Incorrect2" name="question4"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
<?php echo trim(file_get_contents('data/reward'))?><input type="radio" value="Correct" name="question4"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
<?php echo trim(file_get_contents('data/temptation'))?><input type="radio" value="Incorrect3" name="question4"> 
<br />
<br>
 </td>
</tr>
</table>
 
</td>
<td>  
<table height="40%" width="60%" align=right valing=top>
  <col width="50">
  <col width="50">
  <col width="60">
  <col width="60">
  <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> <font size="2"> Choix du voisin </font> </td>
  <tr height="40"> <td></td> <td></td> <td bgcolor="0099CC"> </td> <td bgcolor="FFFF33"> </td> </tr>
  <tr height="60"> <td rowspan=2 align=center> <div class="rotate" size=2 style="font-size:15px;"> <font size="2"> Votre choix </font> </div> </td> 
                   <td bgcolor="0099CC"> </td> <td align=center> <?php echo trim(file_get_contents('data/reward'))?> </td>    
                   <td align=center> <?php echo trim(file_get_contents('data/suckers')) ?> </td>  </tr>  
  <tr height="60"> <td bgcolor="FFFF33"> </td> <td align=center> <?php echo trim(file_get_contents('data/temptation'))?> </td>
                   <td align=center> <?php echo trim(file_get_contents('data/punishment')) ?> </td> </tr>
  <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> Your earn </td> </tr>
</table>

</td>
</tr>
</table>                                                                                       


<br><br>
  Jouez-vous avec les mêmes voisins à chaque round?  <br>
  &nbsp;&nbsp;<input type="radio" value="Correct" name="same_partner"> Oui <br>
  &nbsp;&nbsp;<input type="radio" value="Incorrect" name="same_partner"> Non
  <br><br>

  Les règles du jeu sont-elles les mêmes pour tout le monde?  <br>
  &nbsp;&nbsp;<input type="radio" value="Correct" name="same_rules"> Oui <br>
  &nbsp;&nbsp;<input type="radio" value="Incorrect" name="same_rules"> Non
 
  <br><br>


<?php// $_SESSION['main']=1
?>

<?php
// changing the stage of experiment into the page which checks the answers         
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is now the page which checks the answers       
$_SESSION['step']="checktutorial6".$_SESSION['language'].".php";
?>


Cliquez <input type="submit" value="here" class="btn" name="submit"> pour continuer.
</form>

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->

<?php include_once("html_footer.php"); ?>
