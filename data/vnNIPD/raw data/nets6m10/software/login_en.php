<!-- Login page -->

<head>
<link rel="StyleSheet" href="style.css" type="text/css"> <!-- includes the style shit -->
</head>

<?php include("html_header.php");?>
<br><br><br><br><br><br><br><br>

<center>
<br><br>
Please, fill in the username and password, which are in the envelope we gave to you.
<br>
<br>
<table border="0" width="50%" cellpadding="0" align="center" >
<tr>
<br>
<td width="100%" align="right">

<!-- login form -->
<form name="form" method="post" action="index.php">
<p><label for="username"><b>Username:</b></label>
<input type="text" name="username" /></p>
<p><label for="txtpassword"><b>Password:</b></label>
<input type="password" name="txtpassword" /></p>
</td>   
</tr>   
</table>
<p><input type="submit" name="Submit" value="Submit" class="btn"/></p>
</form>

</center>

<?php include("html_footer.php");?>



