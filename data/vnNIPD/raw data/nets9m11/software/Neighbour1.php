<!-- Prints the action (as background color) and payoff in the previous round of Neighbor 1 -->

<?php
    if(trim(file_get_contents('data/'.substr($usern[0],0,-1).'move'))=='C')
        echo '<table border="1" width="30%" height="40%"  bgcolor="0099CC" >';
        else if(trim(file_get_contents('data/'.substr($usern[0],0,-1).'move'))=='D')
              echo '<table border="1" width="30%" height="40%"  bgcolor="FFFF33" >';
              else echo '<table border="1" width="30%" height="40%"  bgcolor="FFFFFF" >';
?>

<tr><td align=center> <?php echo file_get_contents('data/'.substr($usern[0],0,-1).'score'); ?> </td></tr></table>
                     

