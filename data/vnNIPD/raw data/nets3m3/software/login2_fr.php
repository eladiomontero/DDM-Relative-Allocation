<!-- Login page, if login failed the first time (wrong password or username were typed) -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>

<br><br><br><br><br><br><br><br>

<center>
<br><br>
<p style="color:red"> <b>Une erreur s'est produite.</b> </p>
<br>
Veuillez essayer à nouveau.

<br>
<br>
<table border="0" width="50%" cellpadding="0" align="center" >
<tr>
<br>
<td width="100%" align="right">

<form name="form" method="post" action="index.php">
<p><label for="username"><b>Nom d'utilisateur :</b></label>
<input type="text" name="username" /></p>
<p><label for="txtpassword"><b>Mot de passe :</b></label>
<input type="password" name="txtpassword" /></p>
</td>   
</tr>   
</table>
<p><input type="submit" name="Submit" value="Envoyer" class="btn"/></p>
</form>

</center>



<?php include_once("html_footer.php"); ?>