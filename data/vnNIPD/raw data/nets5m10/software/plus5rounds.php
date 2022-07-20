<!-- Increases the total rounds number by 5 -->
<!-- Warning!!! might not work-->

<?php
$roundsnumber=(int)file_get_contents('data/roundsnumber');
$roundsnumber=$roundsnumber+5;
file_put_contents('data/roundsnumber', (string)$roundsnumber);
?>
<script type="text/javascript">
top.location="index.php";
</script>
