<!--Checks if the answers to the test in tutorial part 8 are correct and presents the answers to the screen -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br>

<?php
$green='#4CC417';
$red='#FF0000';
?>
<b>LES REPONSES</b>

<br><br>
<?php
$user=$_SESSION['username'];
$f1=fopen('../software/data/'.$user.'answers','a+');
fwrite($f1,'question1:'.$_POST['question1']."\n");
fwrite($f1,'question2:'.$_POST['question2']."\n");
fwrite($f1,'question3:'.$_POST['question3']."\n");
fwrite($f1,'question4:'.$_POST['question4']."\n");
fclose($f1);

?>
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
<?php

if($_POST['question1']=='Correct')  
echo "<FONT COLOR=".$green."> Correct. 4 x 5 = 20 </FONT> <br>";      
else        
echo "<FONT COLOR=".$red."> Incorrect. La réponse correcte est: 4 x 5 = 20";            
?>    
<br>
<br>

</td>
<td></td> 
<td align="center">
<img src="pics/question2.png" height="180"> <br>  
<?php
 if($_POST['question2']=='Correct')
     echo "<FONT COLOR=".$green."> Correct. 4 x 1 = 4 </FONT> <br>";
 else
     echo "<FONT COLOR=".$red."> Incorrect. La réponse correcte est: 4 x 1 = 4"; 
?>
<br>
<br>

</td>
</tr>
<tr>
 <td align="center">                 
<img src="pics/question3.png" height="180"> <br>
<?php
  if($_POST['question3']=='Correct')
        echo "<FONT COLOR=".$green."> Correct 2 x 1 + 2 x 6 = 14 </FONT> <br>";
  else 
        echo "<FONT COLOR=".$red."> 
        Incorrect. La réponse correcte est: 2 x 1 + 2 x 6 = 14";  
?>

<br>
<br>
 </td>
<td></td>  
 <td align="center">
<img src="pics/question4.png" height="180"> <br>
<?php
  if($_POST['question4']=='Correct')
        echo "<FONT COLOR=".$green."> Correct. 2 x 0 + 2 x 5 = 10 </FONT> <br>";
  else
        echo "<FONT COLOR=".$red."> Incorrect. La réponse correcte est: 2 x 0 + 2 x 5 = 10";  
        
?>
<br>
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
     <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> <font size="2"> Choix de votre voisin </font> </td>
     <tr height="40"> <td></td> <td></td> <td bgcolor="0099CC"> </td> <td bgcolor="FFFF33"> </td> </tr>
     <tr height="60"> <td rowspan=2 align=center> <div class="rotate" size=2 style="font-size:15px;"> <font size="2"> Votre choix </font> </div> </td>
                      <td bgcolor="0099CC"> </td> <td align=center> <?php echo trim(file_get_contents('data/reward'))?> </td>
                      <td align=center> <?php echo trim(file_get_contents('data/suckers')) ?> </td>  </tr>
     <tr height="60"> <td bgcolor="FFFF33"> </td> <td align=center> <?php echo trim(file_get_contents('data/temptation'))?> </td>
                      <td align=center> <?php echo trim(file_get_contents('data/punishment')) ?> </td> </tr>
     <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> Vous gagnez </td> </tr>
</table>
                                                                           

</td>
</tr>
</table>
<br>
Jouez-vous avec les mêmes voisins à chaque round? <br>
<?php
  if($_POST['same_partner']=='Correct')
        echo "<FONT COLOR=".$green."> Correct. Vos voisins seront les mêmes à chaque round. </FONT> <br>";
  else
        echo "<FONT COLOR=".$red."> Incorrect. Vos voisins seront les mêmes à chaque round. </FONT>";  
        
?>
<br><br>

Les règles du jeu sont-elles les mêmes pour tout le monde? <br>
<?php
  if($_POST['same_rules']=='Correct')
        echo "<FONT COLOR=".$green."> Correct. Les règles du jeu sont les mêmes pour tout le monde. </FONT> <br>";
  else
        echo "<FONT COLOR=".$red."> Incorrect. Les règles du jeu sont les mêmes pour tout le monde. </FONT>";  
        
?>
<br><br>


  

<?php// $_SESSION['main']=1
?>

Cliquez <input type="submit" value="ici" class="btn" name="submit"> pour continuer si toutes les réponses sont correctes ou si vous comprenez vos erreurs. Sinon, levez votre main pour poser des questions.
</form>

<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is now the next page of tutorial       
$_SESSION['step']="part1".$_SESSION['language'].".php";
?>

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->

<?php include_once("html_footer.php"); ?>

