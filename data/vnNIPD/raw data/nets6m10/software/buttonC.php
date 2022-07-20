<!-- The function is used by main_first.php and main_play.php
Draws the button and after pushing the button forwards the value C to the play.php 
which further proceeds the action-->

<form method="post" name="login" action="play.php">
<input type="hidden" name="move" value="C" />
<input type="submit" value="" onclick="clickme()" style="background-color:#0099CC; width:140; height:80">
<input id="timefinalC" type="hidden" name="time" value=""/>
</form>


<!--<script type="text/javascript">
var ran_number=Math.floor(Math.random()*30);
setTimeout('document.login.submit();',ran_number*1000);
</script>-->