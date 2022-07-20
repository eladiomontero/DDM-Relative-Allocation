<!-- Prints the action (as background color) and payoff in the previous round of the player -->

<?php
    if(substr(file_get_contents('data/'.$user.'move'),0,1)=='C')
        echo '<table border="10" width="60%" height="80%"  bgcolor="0099CC" >';
        else if(substr(file_get_contents('data/'.$user.'move'),0,1)=='D')
              echo '<table border="10" width="60%" height="80%"  bgcolor="FFFF33" >';
              else echo '<table border="10" width="60%" height="80%"  bgcolor="FFFFFF" >';
?>


<tr><td align=center> <p style="font-size:30px;"> <?php echo file_get_contents('data/'.$user.'score'); ?></p> </td></tr></table>

