from . import config as cfg
import os.path
from pickle import dump

LINES_PER_JOURNAL = 8

def create_dict(filepath):
    with open(filepath, "rt", encoding=cfg.ENCODING) as journals_file:
        lines = journals_file.readlines()

    line_count = len(lines)
    journal_count = line_count // LINES_PER_JOURNAL
    journal_id_dict = {}
    for idx in range(journal_count):
        start_line = LINES_PER_JOURNAL*idx
        nlmid =      lines[start_line + 7].strip()[7:].strip()
        journal_id = idx + 1
        journal_id_dict[nlmid] = journal_id
    return journal_id_dict

def run(workdir):
    JOURNAL_ID_DICT_FILEPATH = os.path.join(workdir, cfg.JOURNAL_ID_DICT_FILENAME)
    JOURNAL_ID_TXT_FILEPATH = os.path.join(workdir, cfg.JOURNAL_ID_TXT_FILENAME)
    JOURNAL_MEDLINE_FILEPATH = os.path.join(workdir, cfg.JOURNAL_MEDLINE_FILENAME)
  
    journal_id_dict = create_dict(JOURNAL_MEDLINE_FILEPATH)
    with open(JOURNAL_ID_DICT_FILEPATH, "wb") as jid_file:
        dump(journal_id_dict, jid_file)

    with open(JOURNAL_ID_TXT_FILEPATH, "wt") as jit_file:
        for nlmid, journal_id in sorted(journal_id_dict.items(), key=lambda x: x[1]):
            jit_file.write(f"{journal_id}\t{nlmid}\n")
