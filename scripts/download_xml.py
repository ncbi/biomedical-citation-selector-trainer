import gzip
import json
import math
import requests
from time import sleep
import xml.etree.ElementTree as ET


def efetch(webenv, query_key, retmax):
    params = { "db": "pubmed", "WebEnv": webenv, "query_key": query_key, "retmode": "xml", "retstart": 0, "retmax": retmax}
    r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params)
    xml = r.text
    sleep(0.34)
    return xml


def epost(pmid_list):
    pmid_list_str = ",".join([str(pmid) for pmid in pmid_list])
    data = { "id": pmid_list_str}
    r = requests.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/epost.fcgi", data=data)
    
    root_node = ET.fromstring(r.text)
    webenv = root_node.find("WebEnv").text
    query_key = int(root_node.find("QueryKey").text)

    sleep(0.34)

    return webenv, query_key


def run():
    BATCH_SIZE = 5000
    TEST_SET_PATH = "test_set.json.gz"
    XML_PATH_TEMPLATE = "test_set_xml_{}.xml"

    test_set = json.load(gzip.open(TEST_SET_PATH, "rt"))
    test_set_pmids = [e["pmid"] for e in test_set]
    test_set_size = len(test_set_pmids)

    num_batches = math.ceil(test_set_size/BATCH_SIZE)
    for batch_idx in range(num_batches):
        batch_start = batch_idx*BATCH_SIZE
        batch_end = batch_start + BATCH_SIZE
        batch_pmids = test_set_pmids[batch_start:batch_end]

        webenv, query_key = epost(batch_pmids)
        xml = efetch(webenv, query_key, BATCH_SIZE)

        file_number = batch_idx + 1
        save_filepath = XML_PATH_TEMPLATE.format(file_number)
        with open(save_filepath, "wt") as save_file:
            save_file.write(xml)


if __name__ == "__main__":
    run()
