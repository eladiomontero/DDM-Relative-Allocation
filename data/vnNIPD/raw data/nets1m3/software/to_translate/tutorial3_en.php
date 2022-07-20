<!-- Tutorial page 5 -->


<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br>

<b>GAINS PER ROUND </b><br><br>

Durin the experiment you will have to choose between yellow or blue.
In the table below, you can see that the number of points you earn depends on what both you and the other player have chosen. 
<br><br>

<table height="30%" width="50%" align=center>  
   <col width="50">
   <col width="50">  
   <col width="60">  
   <col width="60">
   <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> The other player's choice</td>  
   <tr height="40"> <td></td> <td></td> <td bgcolor="0099CC"> </td> <td bgcolor="FFFF33"> </td> </tr>  
   <tr height="60"> <td rowspan=2 align=center> <div class="rotate" size=2 style="font-size:15px;"> Your choice </div> </td> 
                    <td bgcolor="0099CC"> </td> <td align=center> <?php echo trim(file_get_contents('data/reward'))?> </td>   
                    <td align=center> <?php echo trim(file_get_contents('data/suckers')) ?> </td>  </tr>  
   <tr height="60"> <td bgcolor="FFFF33"> </td> <td align=center> <?php echo trim(file_get_contents('data/temptation'))?> </td>  
                    <td align=center> <?php echo trim(file_get_contents('data/punishment')) ?> </td> </tr>
   <tr height="40"> <td></td> <td></td> <td colspan=2 align=center> Your earn </td> </tr>                   
</table>    

<br>
Therefore <br>
<ul>
<li> If you choose <span style="background-color: #0099CC">BLUE</span> 
and the other player chooses <span style="background-color: #0099CC">BLUE</span> 
you earn <?php echo trim(file_get_contents('data/reward'))?> 
and the other player earns <?php echo trim(file_get_contents('data/reward'))?>  </li> 
<br>
<li> If you choose <span style="background-color: #0099CC">BLUE</span> 
and the other player chooses <span style="background-color: #FFFF33">YELLOW</span> 
you earn <?php echo trim(file_get_contents('data/suckers'))?> 
and the other player earns <?php echo trim(file_get_contents('data/temptation'))?>  </li>
<br>
<li> If you choose <span style="background-color: #FFFF33">YELLOW</span> 
and the other player chooses <span style="background-color: #0099CC">BLUE</span> 
you earn <?php echo trim(file_get_contents('data/temptation'))?> 
and the other player earns <?php echo trim(file_get_contents('data/suckers'))?>  </li>
<br>
<li> If you choose <span style="background-color: #FFFF33">YELLOW</span> 
and the other player chooses <span style="background-color: #FFFF33">YELLOW</span> 
you earn <?php echo trim(file_get_contents('data/punishment'))?> 
and the other player earns <?php echo trim(file_get_contents('data/punishment'))?>  </li>
<br>
<br>

Each point you gain corresponds to <?php echo trim(file_get_contents('data/pointvalue'))?> euros.
This table will be also be visibile in the top right corner during every step of the experiment. 
You can also find it in the written instructions.

<?php
// changing the stage of experiment into the next page of tutorial            
// this way after pressing the button, user will be redirected to index page   
// and then to the next stage of experiment which is next page of tutorial       
$_SESSION['step']="tutorial4".$_SESSION['language'].".php";
?>
  
<form name="form1" method="post" action="index.php">
Click <input type="submit" value="here" class="btn" name="submit"> to continue.
</form><br />

<!--<script type="text/javascript">
setTimeout('document.form1.submit()',1000);
</script>-->


<?php include_once("html_footer.php"); ?>