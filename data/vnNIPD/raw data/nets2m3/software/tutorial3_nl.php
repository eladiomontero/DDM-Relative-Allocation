<!-- Tutorial page 5 -->


<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br>

<b>OPBRENGST PER RONDE </b><br><br>

Tijdens het experiment moet je een keuze maken tussen 2 acties: geel of blauw.  Afhankelijk van de kleur die jij kiest en die van je 4 buren in het rooster zal je een zeker aantal punten verdienen.  In de tabel hieronder zie je de punten die je kan verdienen in relatie tot de actie van elke buur in het rooster. 

<br><br>

<table height="30%" width="50%" align=center>  
   <col width="50">
   <col width="50">  
   <col width="60">  
   <col width="60">
   <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> De keuze van de buren</td>  
   <tr height="40"> <td></td> <td></td> <td bgcolor="0099CC"> </td> <td bgcolor="FFFF33"> </td> </tr>  
   <tr height="60"> <td rowspan=2 align=center> <div class="rotate" size=2 style="font-size:15px;"> Jou Keuze</div> </td> 
                    <td bgcolor="0099CC"> </td> <td align=center> <?php echo trim(file_get_contents('data/reward'))?> </td>   
                    <td align=center> <?php echo trim(file_get_contents('data/suckers')) ?> </td>  </tr>  
   <tr height="60"> <td bgcolor="FFFF33"> </td> <td align=center> <?php echo trim(file_get_contents('data/temptation'))?> </td>  
                    <td align=center> <?php echo trim(file_get_contents('data/punishment')) ?> </td> </tr>
   <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> Jij verdient </td> </tr>                   
</table>    

<br>
Daarom <br>
<ul>
<li> Als jij <span style="background-color: #0099CC">BLAUW</span> kiest <br>
terwijl drie van je buren  <span style="background-color: #0099CC">BLAUW</span> kiezen <br>
en de laaste buur  <span style="background-color: #FFFF33">GEEL</span> kiezen <br>
dan verdien je 3 &#215; <?php echo trim(file_get_contents('data/reward'))?> + 1 &#215; <?php echo trim(file_get_contents('data/suckers'))?> = 15 
</li> 
<br>

<li> Als jij <span style="background-color: #FFFF33">GEEL</span> kiest <br>
terwijl twee buren <span style="background-color: #0099CC">BLAUW</span> kiest  <br>
en de twee andere buren kiezen  <span style="background-color: #FFFF33">GEEL</span> <br>
dan verdien jij 2 &#215; <?php echo trim(file_get_contents('data/temptation'))?> + 2 &#215; <?php echo trim(file_get_contents('data/punishment'))?> =  14 
</li> 
<br>
<br>

Elk punt dat jij verdient stemt overeen met  <?php echo round(trim(file_get_contents('data/pointvalue')),2)?> euro.
De tabel met punten die je kan verdienen zal ook worden getoond in de linkerbovenhoek tijdens het experiment.  Je kan de tabel ook terugvinden in de geprinte instructies.

<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial4".$_SESSION['language'].".php";
?>
  
<form name="form1" method="post" action="index.php">
Druk <input type="submit" value="hier" class="btn" name="submit"> om verder te gaan.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->


<?php include_once("html_footer.php"); ?>
