<?php
$zahl = $_POST["zahl"];
$id = $_GET["id"];
$output = shell_exec("C:\Users\JulianEbeling\AppData\Local\Programs\Python\Python39\python.exe ki.py run $id $zahl 2>&1");
echo $output;
header("Location: tester.php?formel=$id&result=$output")
?>