<head>
<link rel="StyleSheet" href="style.css" type="text/css">
</head>

<!-- logo -->
<?php include_once("html_header.php"); ?>
<br><br><br><br><br><br><br><br>

<?php
  if(file_get_contents('data/finished')=="finished"){
    $_SESSION['step']="last".$_SESSION['language'].".php";    
  }
?>  
            
Gelieve een ogenblik te wachten, bedankt...

<br>
<br>

                   
<script type="text/javascript">
setInterval('top.location="index.php";',1000);
</script>
                   
<?php include_once("html_footer.php"); ?>