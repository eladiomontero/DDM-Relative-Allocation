<!-- Tutorial page 5 -->


<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br>

<b>GAINS PAR ROUND </b><br><br>

Durant l'expérience vous devrez choisir: jaune ou bleu. En fonction de la couleur que vous et les autres participants choisissez, vous gagnerez un nombre différent de points. 
Dans la table ci-dessous, vous voyez les nombres de points que vous pouvez gagner dans les interactions avec CHACUN de vos voisins, en fonction de ce que vous faites et de ce qu'ils font. 
<br><br>

<table height="30%" width="50%" align=center>  
   <col width="50">
   <col width="50">  
   <col width="60">  
   <col width="60">
   <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> Choix du voisin</td>  
   <tr height="40"> <td></td> <td></td> <td bgcolor="0099CC"> </td> <td bgcolor="FFFF33"> </td> </tr>  
   <tr height="60"> <td rowspan=2 align=center> <div class="rotate" size=2 style="font-size:15px;"> Votre choix </div> </td> 
                    <td bgcolor="0099CC"> </td> <td align=center> <?php echo trim(file_get_contents('data/reward'))?> </td>   
                    <td align=center> <?php echo trim(file_get_contents('data/suckers')) ?> </td>  </tr>  
   <tr height="60"> <td bgcolor="FFFF33"> </td> <td align=center> <?php echo trim(file_get_contents('data/temptation'))?> </td>  
                    <td align=center> <?php echo trim(file_get_contents('data/punishment')) ?> </td> </tr>
   <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> Vous gagnez </td> </tr>                   
</table>    

<br>
Donc <br>
<ul>
<li> Si vous choisissez <span style="background-color: #0099CC">BLEU</span>  <br>
et 3 de vos voisins choisissent <span style="background-color: #0099CC">BLEU</span> <br>
et 1 de vos voisins choisissent  <span style="background-color: #FFFF33">JAUNE</span> <br>
vous gagnez 3 &#215; <?php echo trim(file_get_contents('data/reward'))?> + 1 &#215; <?php echo trim(file_get_contents('data/suckers'))?> = 15 
</li> 
<br>

<li> Si vous choisissez <span style="background-color: #FFFF33">JAUNE</span>  <br>
et 2 de vos voisin choisit <span style="background-color: #0099CC">BLEU</span> <br>
et 2 de vos voisins choisissent  <span style="background-color: #FFFF33">JAUNE</span> <br>
vous gagnez 2 &#215; <?php echo trim(file_get_contents('data/temptation'))?> + 2 &#215; <?php echo trim(file_get_contents('data/punishment'))?> = 14 
</li> 
<br>
<br>

Chaque point que vous gagnez correspond à <?php echo round(trim(file_get_contents('data/pointvalue')),2)?> euros.
La table avec les points pour chaque choix sera aussi visible dans le coin supérieur à gauche à chaque étape de l'expérience. 
Vous pouvez aussi la trouver dans les instructions écrites. 

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
