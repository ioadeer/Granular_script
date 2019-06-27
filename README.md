# Tesis de corpus based concatenative synthesis

En este repo hay un directorios que poseen un archivo wav, archivos csv resultantes del análisis de features de este audio y un script para reproducirlo. El script se usa:

python3 player.py -f 01\_mono.wav -c 01\_mono\_frameSize\_16384\_hopSize\_4096.csv

Es redundante que haya un script por directorio y poco práctico, pero todavía no agregue que el script parsee el path de archivo que toma como  input. Para no tener todos los archivos csv y audios en un mismo directorio, los separé y copié el script a cada uno de ellos. Esto era lo que te comentaba que me parecía que no estaba bien resuelto. De todas formas, cuando lo mejore, la idea es usar un solo script.

En el script se extraen del archivo csv cierto features, de forma arbitraria para hacer análisis de PCA. 
