# Provide a list of tar files to be extracted and processed - deleting intermediate steps along the way
# Usage: SOURCE TARGET METADATA


# Arguments: $1 tarfile $2 target $3 metadata
gaml_processtar () {
	tarfile=$(readlink -f "$1")

	echo "Begin" "$tarfile"

	temp=$(readlink -m "$2/../temp/"$(basename "$tarfile" .tar))

	echo $tarfile
	echo $temp

	#java -jar TarCrawler.jar "$tarfile" "$temp" ".tex,.cls,.sty,.tab"
	#python3 findTexDocuments.py "$temp" "$2" -s astro -m "$3"
	#rm -r "$temp"
	
	echo "End" "$tarfile"
}



export -f gaml_processtar
ls "$1"/*.tar -1 | parallel --no-notice -P 8 -N 1 gaml_processtar {1} "$2" "$3"
