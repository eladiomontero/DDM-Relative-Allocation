<!-- Tutorial page 5 -->


<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br>

<b>OPBRENGST PER RONDE </b><br><br>

Tijdens het experiment moet je kiezen tussen geel en blauw.
In de tabel hieronder kan je zien dat de punten die je krijgt in een ronde afhankelijk zijn van zowel jou keuze als die van de andere deelnemer.

<br><br>

<table height="30%" width="50%" align=center>  
   <col width="50">
   <col width="50">  
   <col width="60">  
   <col width="60">
   <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> De keuze van de andere speler</td>  
   <tr height="40"> <td></td> <td></td> <td bgcolor="0099CC"> </td> <td bgcolor="FFFF33"> </td> </tr>  
   <tr height="60"> <td rowspan=2 align=center> <div class="rotate" size=2 style="font-size:15px;"> Jou keuze </div> </td> 
                    <td bgcolor="0099CC"> </td> <td align=center> <?php echo trim(file_get_contents('data/reward'))?> </td>   
                    <td align=center> <?php echo trim(file_get_contents('data/suckers')) ?> </td>  </tr>  
   <tr height="60"> <td bgcolor="FFFF33"> </td> <td align=center> <?php echo trim(file_get_contents('data/temptation'))?> </td>  
                    <td align=center> <?php echo trim(file_get_contents('data/punishment')) ?> </td> </tr>
   <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> You earn </td> </tr>                   
</table>    

<br>
Dus <br>
<ul>
<li> Als je <span style="background-color: #0099CC">BLUE</span> kiest
en de andere speler kiest <span style="background-color: #0099CC">BLUE</span> 
dan verdien je <?php echo trim(file_get_contents('data/reward'))?> punten en de
andere speler verdient <?php echo trim(file_get_contents('data/reward'))?>  </li> 
<br>
<li> Als je <span style="background-color: #0099CC">BLUE</span> kiest
en de andere speler kiest <span style="background-color: #FFFF33">YELLOW</span> 
dan verdien je <?php echo trim(file_get_contents('data/suckers'))?> punten en de
andere speler verdient  <?php echo trim(file_get_contents('data/temptation'))?>  </li>
<br>
<li> Als je <span style="background-color: #FFFF33">YELLOW</span> kiest
en de andere speler kiest  <span style="background-color: #0099CC">BLUE</span> 
dan verdien je <?php echo trim(file_get_contents('data/temptation'))?> punten en de
andere speler verdient  <?php echo trim(file_get_contents('data/suckers'))?>  </li>
<br>
<li> Als je  <span style="background-color: #FFFF33">YELLOW</span> kiest
en de andere speler kiest <span style="background-color: #FFFF33">YELLOW</span> 
dan verdien je <?php echo trim(file_get_contents('data/punishment'))?> punten en de
andere speler verdient <?php echo trim(file_get_contents('data/punishment'))?>  </li>
<br>
<br>

Elk punt dat je verdient komet overeen met <?php echo trim(file_get_contents('data/pointvalue'))?> euro.

Deze tabel zal ook zichtbaar zijn in de rechterbovenhoek van elke stap van het experiment.
Je kan de tabel ook vinden in de uitleg op het blad.

<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial4".$_SESSION['language'].".php";
?>
  
<form name="form1" method="post" action="index.php">
Klik <input type="submit" value="here" class="btn" name="submit"> om verder te gaan.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->


<?php include_once("html_footer.php"); ?>