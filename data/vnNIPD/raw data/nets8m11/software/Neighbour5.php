<!-- Prints the action (as background color) and payoff in the previous round of Neighbor 5 -->


<?php
    if(trim(file_get_contents('data/'.substr($usern[4],0,-1).'move'))=='C')
        echo '<table border="1" width="40%" height="40%"  bgcolor="0099CC" >';
        else if(trim(file_get_contents('data/'.substr($usern[4],0,-1).'move'))=='D')
              echo '<table border="1" width="40%" height="40%"  bgcolor="FFFF33" >';
?>

<tr><td align=center><?php echo file_get_contents('data/'.substr($usern[4],0,-1).'score'); ?> </td></tr></table> 
