<?php
session_start();
$upload_folder = '../upload/'; //Das Upload-Verzeichnis
$filename = pathinfo($_FILES['datei']['name'], PATHINFO_FILENAME);
$extension = strtolower(pathinfo($_FILES['datei']['name'], PATHINFO_EXTENSION));
 
 
//Überprüfung der Dateiendung
$allowed_extensions = array('png', 'jpg', 'jpeg', 'gif');
if(!in_array($extension, $allowed_extensions)) {
	echo "<script type='text/javascript'>alert('Nur png, jpg, jpeg und gif-Dateien sind erlaubt.');</script>";
	echo "<meta http-equiv='refresh' content='0;URL=forumprofile.php?showuser=$username' />";
}else{
	//Überprüfung der Dateigröße
$max_size = 5000*1024; //500 KB
if($_FILES['datei']['size'] > $max_size) {
	echo "<script type='text/javascript'>alert('Die Maximale Dateigröße beträgt 1MB');</script>";
	echo "<meta http-equiv='refresh' content='0;URL=forumprofile.php?showuser=$username' />";
}else{
 
//Pfad zum Upload
$new_path = $upload_folder.$filename.'.'.$extension;
 
//Neuer Dateiname falls die Datei bereits existiert
if(file_exists($new_path)) { //Falls Datei existiert, hänge eine Zahl an den Dateinamen
 $id = 1;
 do {
 $new_path = $upload_folder.$filename.'_'.$id.'.'.$extension;
 $id++;
 } while(file_exists($new_path));
}
move_uploaded_file($_FILES['datei']['tmp_name'], $new_path);
echo "<script type='text/javascript'>alert('Das Bild wurde hochgeladen.');</script>";
echo "<meta http-equiv='refresh' content='0;URL=bild.php?showimage=$new_path' />";
}
}
 
 

 

?>