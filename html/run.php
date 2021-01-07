<?php
include("config.php");
$zahl = $_POST["zahl"];
$id = $_GET["id"];
$output = shell_exec("$python_path ki.py run $id $zahl 2>&1");
echo $output;
header("Location: tester.php?formel=$id&result=$output")
?>