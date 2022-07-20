<!--Checks if the daemon is started-->

<?php  

echo "prooooooba!";
if (trim(file_get_contents('data/deamonstarted'))=="started"){ 
    $_SESSION['step']="login".$_SESSION['language'].".php";
    echo "proba";
?>

<script type="text/javascript">
top.location="index.php";
</script>

<?php    
  echo "proba2"
}
else {
  echo "proba3";
  include_once("notstarted".$_SESSION['language'].".php");
}
?>
