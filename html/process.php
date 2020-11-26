<?php
$formel = $_POST["formel"];
$output = shell_exec("C:\Users\JulianEbeling\AppData\Local\Programs\Python\Python39\python.exe ki.py train $formel 2>&1");
echo $output;
header("Location: tester.php?formel=$output")
?>