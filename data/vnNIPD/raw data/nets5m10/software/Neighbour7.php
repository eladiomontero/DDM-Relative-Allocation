<!-- Prints the action (as background color) and payoff in the previous round of Neighbor 7 -->


<?php
    if(substr(file_get_contents('data/'.substr($usern[6],0,-1).'move'),0,1)=='C')
        echo '<table border="1" width="40%" height="40%"  bgcolor="0099CC" >';
        else if(substr(file_get_contents('data/'.substr($usern[6],0,-1).'move'),0,1)=='D')
              echo '<table border="1" width="40%" height="40%"  bgcolor="FFFF33" >';
?>

<tr><td align=center> <?php echo file_get_contents('data/'.substr($usern[6],0,-1).'score'); ?> </td></tr></table> 
