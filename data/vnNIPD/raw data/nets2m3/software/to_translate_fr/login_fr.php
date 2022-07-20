<!-- Login page -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css"> <!-- includes the style shit -->
</head>

<?php include("html_header.php");?>
<br><br><br><br><br><br><br><br>

<center>
<br><br>
Veuillez compléter le nom d'utilisateur et le mot de passe qui se trouvent dans l'enveloppe que nous vous avons donnée.
<br>
<br>
<table border="0" width="50%" cellpadding="0" align="center" >
<tr>
<br>
<td width="100%" align="right">

<!-- login form -->
<form name="form" method="post" action="index.php">
<p><label for="username"><b>Nom d'utilisateur :</b></label>
<input type="text" name="username" /></p>
<p><label for="txtpassword"><b>Mot de passe :</b></label>
<input type="password" name="txtpassword" /></p>
</td>   
</tr>   
</table>
<p><input type="submit" name="Submit" value="Submit" class="btn"/></p>
</form>

</center>

<?php include("html_footer.php");?>



