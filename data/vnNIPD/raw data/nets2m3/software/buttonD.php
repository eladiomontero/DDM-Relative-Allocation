<!-- Draws the button and after pushing the button forwards the value D to the play.php -->

<form method="post" name="login" action="play.php">
<input type="hidden" name="move" value="D" />
<input type="submit" value="" onclick="clickme()" style="background-color:#FFFF33; width:140; height:80">
<input id="timefinalD" type="hidden" name="time" value=""/>
</form>




<!--<script type="text/javascript">
var ran_number=Math.floor(Math.random()*30);
setTimeout('document.login.submit();',ran_number*1000);
</script>-->