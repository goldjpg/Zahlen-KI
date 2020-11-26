<!DOCTYPE html>
<html lang="de">
<head>
	<meta charset="UTF-8">
	<title>Objekt-KI</title>
	<link rel="stylesheet" href="../css/formate.css">
</head>
<body>
	<header>
		<p>Objekt-KI</p>
	</header>
	
	<article>
		<a href="bild.php">[Reset]</a>
		<h1>Zahl eingeben</h1>
		<br>
		<main>
		<form action="run.php?id=<?php echo $_GET["formel"]?>" method="post" enctype="multipart/form-data">
			<input type="number" name="zahl">
			<input type="submit" value="Trainieren">
		</form>
		<br>
		<h2>Ergebnis: <?php echo @$_GET["result"]?><h2>
	</main>
	</article>
	<footer>
		By Jan, Arne und Julian 2020
	</footer>
</body>
</html>