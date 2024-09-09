if __name__ == '__main__':

	from gaml.utilities.argparseactions import ArgumentParser,FileAction

	import dominate
	from dominate.tags import div,link,h1,h2,table,thead,tbody,tr,th,td,a

	import pandas

	parser = ArgumentParser(description='Generate webpage for annotation sample from annotators listings.')
	parser.add_argument('summary',action=FileAction,mustexist=True,help='Summary file for annotation sample.')
	parser.add_argument('output',action=FileAction,mustexist=False,help='Output location for HTML file.')
	parser.add_argument('name',help='Sample name.')
	args = parser.parse_args()

	doc = dominate.document(title='Numerical Atlas Annotation Project')

	initialdata = pandas.read_csv(args.summary)
	data = pandas.concat([
		initialdata,
		initialdata.rename(columns={'annotator1':'annotator2','annotator2':'annotator1'},copy=True)],sort=False)

	with doc.head:
		link(rel="stylesheet",
				href="https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css",
				integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS",
				crossorigin="anonymous")

	def arXiv_link(arXiv_id):
		return 'https://arxiv.org/abs/' + arXiv_id

	def numericalatlas_link(annotator,arXiv_id):
		return 'http://numericalatlas.cs.ucl.ac.uk/annotation/' + annotator + '/#/' + args.name + '/' + arXiv_id.replace('/','')

	with doc:
		with div(cls='container'):
			div(div(h1('Numerical Atlas'),cls='col-12'),cls='row')
			div(div('Listings for: ' + args.name,cls='col-12 pl-5 lead'),cls='row')

			for name,group in data.groupby('annotator1'):
				with div(cls='row mt-5'):
					div(cls='row').add(h2(name,id=name))
				with div(cls='row mt-3'):
					with table(cls='table table-hover'):
						with thead(cls="thead-light").add(tr()):
							th('arXiv',scope="col")
							th('Annotator 1',scope="col")
							th('Annotator 2',scope="col")
						with tbody():
							for i,row in group.sort_values('data').iterrows():
								with tr():
									td().add(a(row['data'],
											href=arXiv_link(row['data'])))
									td().add(a(name,
											href=numericalatlas_link(name,row['data'])))
									td().add(a(row['annotator2'],
											href=numericalatlas_link(row['annotator2'],row['data'])))

	with open(args.output,'w') as f:
		f.write(str(doc))

