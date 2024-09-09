#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility to download the metadata for every paper on the arXiv.
Written by: Dan Foreman-Mackey from https://github.com/dfm/data.arxiv.io as scrape.py
Modified by: Tom Crossland

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["download"]

import re
import time
import logging
import requests
import xml.etree.cElementTree as ET
import dateutil.parser

# Download constants
resume_re = re.compile(r".*<resumptionToken.*?>(.*?)</resumptionToken>.*")
url = "http://export.arxiv.org/oai2"

# Parse constant
record_tag = ".//{http://www.openarchives.org/OAI/2.0/}record"
format_tag = lambda t: ".//{http://arxiv.org/OAI/arXiv/}" + t
date_fmt = "%a, %d %b %Y %H:%M:%S %Z"


def download(start_date=None, max_tries=10):
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv"}
    if start_date is not None:
        params["from"] = start_date

    failures = 0
    while True:
        # Send the request.
        r = requests.post(url, data=params)
        code = r.status_code

        # Asked to retry
        if code == 503:
            to = int(r.headers["retry-after"])
            logging.info("Got 503. Retrying after {0:d} seconds.".format(to))

            time.sleep(to)
            failures += 1
            if failures >= max_tries:
                logging.warn("Failed too many times...")
                break

        elif code == 200:
            failures = 0

            # Write the response to a file.
            content = r.text
            yield from parse(content)

            # Look for a resumption token.
            token = resume_re.search(content)
            #break ############################################################### TEMPORARY LINE - ONLY WANT FIRST BATCH HERE
            if token is None:
                break
            token = token.groups()[0]

            # If there isn't one, we're all done.
            if token == "":
                logging.info("All done.")
                break

            logging.info("Resumption token: {0}.".format(token))

            # If there is a resumption token, rebuild the request.
            params = {"verb": "ListRecords", "resumptionToken": token}

            # Pause so as not to get banned.
            to = 15
            logging.info("Sleeping for {0:d} seconds so as not to get banned."
                         .format(to))
            time.sleep(to)

        else:
            # Wha happen'?
            r.raise_for_status()


def parse(xml_data):
    tree = ET.fromstring(xml_data)
    #results = []
    for i, r in enumerate(tree.findall(record_tag)):
        try:
            arxiv_id = r.find(format_tag("id")).text
            date = dateutil.parser.parse(r.find(format_tag("created")).text)
            title = r.find(format_tag("title")).text
            abstract = r.find(format_tag("abstract")).text
            categories = r.find(format_tag("categories")).text
        except:
            logging.error("Parsing of record failed:\n{0}".format(r))
        else:
            yield arxiv_id, date, title, abstract, categories
    #return results




if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)

	import argparse
	import pickle
	from gaml.utilities.argparseactions import FileAction

	parser = argparse.ArgumentParser(description="Download all arXiv metadata and store in a Whoosh index.")
	parser.add_argument("metadatapath",action=FileAction, mustexist=False, findavailable=True, help='Path in which to store metadata file).')
	args = parser.parse_args()

	metadata = {}

	for arxiv_id, date, title, abstract, categories in download():
		metadata[arxiv_id] = {'date': date, 'title': title, 'categories': categories}

	with open(args.metadatapath, 'wb') as f:
		# Pickle the metadata dictionary using the highest protocol available.
		pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
