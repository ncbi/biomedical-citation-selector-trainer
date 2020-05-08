from . import config as cfg
import gzip
from .helper import create_dir, load_indexing_periods
import json
import math
import os
import requests
from time import sleep


def efetch(webenv, query_key, retstart):
    #https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&webenv=NCID_1_195274084_130.14.22.33_9001_1587464603_439519324_0MetA0_S_MegaStore&query_key=1&retmode=xml&retstart=0&retmax=10
    params = {"db": "pubmed", "WebEnv": webenv, "query_key": query_key, "retmode": "xml", "retstart": retstart, "retmax": cfg.EUTILS_RETMAX}
    r = requests.get(cfg.EFETCH_URL, params=params)
    xml = r.text
    sleep(cfg.EUTILS_DELAY)
    return xml


def esearch(nlmid, min_year, max_year):
    #https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=0410462[nlmid]&mindate=2018&maxdate=2020&datetype=pdat&retmode=json&usehistory=y

    params = {"db": "pubmed", "term": f"{nlmid}[nlmid]", "mindate": f"{min_year:04d}", "maxdate": f"{max_year:04d}", "datetype": "pdat", "retmode": "json", "usehistory": "y"}
    r = requests.get(cfg.ESEARCH_URL, params=params)
    
    esearchresult = json.loads(r.text)["esearchresult"]
    webenv = esearchresult["webenv"]
    query_key = esearchresult["querykey"]
    num_results = int(esearchresult["count"])
 
    sleep(cfg.EUTILS_DELAY)

    return webenv, query_key, num_results


def run(workdir):

    datadir = os.path.join(workdir, cfg.MEDLINE_DATA_DIR)
    create_dir(datadir)

    downloaded_data_filepath_template = os.path.join(datadir, cfg.DOWNLOADED_DATA_FILENAME_TEMPLATE)
    selective_indexing_periods_filepath = os.path.join(workdir, cfg.SELECTIVE_INDEXING_PERIODS_FILENAME)
    
    indexing_periods = load_indexing_periods(selective_indexing_periods_filepath, cfg.ENCODING, False)

    count = 0
    num_journals = len(indexing_periods)
    for idx, nlmid in enumerate(sorted(indexing_periods)):
        print(f"{idx + 1}/{num_journals}", end="\r")    
        for period in indexing_periods[nlmid]:
            start_year = period["start_year"]
            end_year = period["end_year"]

            min_year = (start_year + 1)
            max_year = (end_year - 1) if end_year is not None else cfg.MODEL_MAX_DATE.year

            webenv, query_key, num_results = esearch(nlmid, min_year, max_year)
        
            batch_size = cfg.EUTILS_RETMAX
            num_batches = math.ceil(num_results/batch_size)
            for batch_idx in range(num_batches):
                count += 1
                retstart = batch_idx*batch_size
                xml = efetch(webenv, query_key, retstart)
                save_filepath = downloaded_data_filepath_template.format(count)
                with gzip.open(save_filepath, "wt", encoding=cfg.ENCODING) as save_file:
                    save_file.write(xml)

    print(f"{num_journals}/{num_journals}")
    return count