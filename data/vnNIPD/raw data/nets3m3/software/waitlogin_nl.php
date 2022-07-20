<!--Checks if all users are logged in, by checking the content of file "loggedin", 
where background process will write "loggedin", when all the players are loggedin -->


<head>
<!--<link rel="StyleSheet" href="styleproba.css" type="text/css">-->
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<?php
  if(file_get_contents('data/allloggedin')=="allloggedin")
  $_SESSION['step']="tutorial1".$_SESSION['language'].".php";
?>  
            
Gelieve te wachten tot alle andere deelnemers zijn ingelogd, bedankt.
<br>
<br>

                   
<script type="text/javascript">
setInterval('top.location="index.php";',1000);
</script>
                   
<?php include_once("html_footer.php"); ?>                   