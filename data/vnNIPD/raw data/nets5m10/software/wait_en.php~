<!--Checks if all users are ready to play, by checking the content of file "started", 
where daemon will write "started", when all the users are ready -->


<head>
<!--<link rel="StyleSheet" href="styleproba.css" type="text/css">-->
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<?php
  if(file_get_contents('data/started')=="started") 
  $_SESSION['step']="main.php";
?>  
            
<?php echo $_SESSION['username']." ";?>
please, wait until the other participant read the tutorial. Thank you...
<br>
<br>

<script type="text/javascript">
setInterval('top.location="index.php";',1000);
</script>

<?php include_once("html_footer.php"); ?>                   