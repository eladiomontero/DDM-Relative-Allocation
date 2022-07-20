<!-- Tutorial page 5 -->


<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br>

<b>GAINS PAR TOUR </b><br><br>

Durant l'expérience, vous devrez choisir entre bleu ou jaune.
Dans la table ci-dessous, vous pouvez voir que le nombre de points que vous gagnez dépendra de ce que vous et l'autre participants choisirez. 
<br><br>

<table height="30%" width="50%" align=center>  
   <col width="50">
   <col width="50">  
   <col width="60">  
   <col width="60">
   <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> Le choix de l'autre joueur</td>  
   <tr height="40"> <td></td> <td></td> <td bgcolor="0099CC"> </td> <td bgcolor="FFFF33"> </td> </tr>  
   <tr height="60"> <td rowspan=2 align=center> <div class="rotate" size=2 style="font-size:15px;"> Votre choix </div> </td> 
                    <td bgcolor="0099CC"> </td> <td align=center> <?php echo trim(file_get_contents('data/reward'))?> </td>   
                    <td align=center> <?php echo trim(file_get_contents('data/suckers')) ?> </td>  </tr>  
   <tr height="60"> <td bgcolor="FFFF33"> </td> <td align=center> <?php echo trim(file_get_contents('data/temptation'))?> </td>  
                    <td align=center> <?php echo trim(file_get_contents('data/punishment')) ?> </td> </tr>
   <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> Vous gagnez </td> </tr>                   
</table>    

<br>
Therefore <br>
<ul>
<li> Si vous choisissez <span style="background-color: #0099CC">BLEU</span> 
et que l'autre joueur choisit <span style="background-color: #0099CC">BLEU</span> 
vous gagnez <?php echo trim(file_get_contents('data/reward'))?> 
et l'autre joueur gagne <?php echo trim(file_get_contents('data/reward'))?>  </li> 
<br>
<li> Si vous choisissez <span style="background-color: #0099CC">BLEU</span> 
et que l'autre joueur choisit <span style="background-color: #FFFF33">JAUNE</span> 
vous gagnez <?php echo trim(file_get_contents('data/suckers'))?> 
et l'autre joueur gagne <?php echo trim(file_get_contents('data/temptation'))?>  </li>
<br>
<li> Si vous choisissez <span style="background-color: #FFFF33">JAUNE</span> 
et que l'autre joueur choisit <span style="background-color: #0099CC">BLEU</span> 
vous gagnez <?php echo trim(file_get_contents('data/temptation'))?> 
et l'autre joueur gagne <?php echo trim(file_get_contents('data/suckers'))?>  </li>
<br>
<li> Si vous choisissez <span style="background-color: #FFFF33">JAUNE</span> 
et que l'autre joueur choisit <span style="background-color: #FFFF33">JAUNE</span> 
vous gagnez <?php echo trim(file_get_contents('data/punishment'))?> 
et l'autre joueur gagne <?php echo trim(file_get_contents('data/punishment'))?>  </li>
<br>
<br>

Chaque point que vous gagnez correspond à <?php echo trim(file_get_contents('data/pointvalue'))?> euros.
Cette table restera visible dans le coin supérieur droit de chaque étape de l'expérience. 
Vous la retrouverez aussi dans les instructions écrites.

<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial4".$_SESSION['language'].".php";
?>
  
<form name="form1" method="post" action="index.php">
Cliquez <input type="submit" value="ici" class="btn" name="submit"> pour continuer.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->


<?php include_once("html_footer.php"); ?>