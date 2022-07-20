<?php  
  session_start();
  unset($_SESSION['username']);
?>
<script type="text/javascript">
top.location="index.php";
</script>

<!--Logout function for one player, used during the test phase -->