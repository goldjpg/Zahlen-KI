<?php
include("config.php");
$formel = $_POST["formel"];
$output = shell_exec("$python_path ki.py train $formel 2>&1");
echo $output;
header("Location: tester.php?formel=$output")
?>