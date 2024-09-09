This library requires that LaTeXML be installed and accessible on the system path (may be installed using apt-get install latexml). The gaml directory must also be accessible from the Python installation.

A .tex file may be opened with the following:

```
import preprocessing.latexmlpy as latexml
texfile = 'path/to/tex/file.tex'
doc = latexml.opentex(texfile, timeout=300)
```
The document will now be opened as an lxml ElementTree and the method will return the root of this tree. The parsing attempt will timeout after timeout seconds (default 5 minutes), and if timeout occurs, or the parsing otherwise fails, the method returns None.

The .tex file should be located in a directory with any .sty and .cls files required for compilation. Note: The file may still parse without these files, but the results will likely include unusual errors.

If the file opens correctly, elements with a given tag may be extracted using:

```
doc.findall('.//tag')
```

Tables may be extracted using the following:

```
from preprocessing.tables import process_table
for elem in doc.findall('.//table'):
	table,borders = process_table(elem)
	print(table)
	print(borders)
```
Table processing is still rudimentary at this time.
