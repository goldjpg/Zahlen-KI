<!DOCTYPE html>
<html lang="de">
<head>
	<meta charset="UTF-8">
	<title>Objekt-KI</title>
	<link rel="stylesheet" href="../css/formate.css">
	<?php
		@$bildurl = $_GET["showimage"];
	?>
</head>
<body>
	<header>
		<p>Objekt-KI</p>
	</header>
	
	<article>
		<h1>Bild hochladen</h1>
		<br>
		<main>
		<form action="upload.php" method="post" enctype="multipart/form-data">
			<input type="file" name="datei">
			<input type="submit" value="Hochladen">
		</form>
		<?php
			if($bildurl != null){
				echo "<br><img src='$bildurl' width=100px height=100px>";
			    echo "<br><a href='process.php?imageurl=$bildurl'>Process</a>";
			}			
		?>
	</main>
	</article>
	<footer>
		By Jan, Arne und Julian 2020
	</footer>
</body>
</html>