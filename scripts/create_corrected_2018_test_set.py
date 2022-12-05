import json
import gzip
import random

selectively_indexed_journals_file = "selectively_indexed_id_mapping_29th_Mar_21.json"
train_set_file  = "train_set_all_data.json.gz"
test_set_file = "test_set_2018_corrected.json.gz"

selectively_indexed_journals = set(json.load(open(selectively_indexed_journals_file)))

train_set = json.load(gzip.open(train_set_file, "rt", encoding="utf8"))
print(f"Train set len :{len(train_set)}")

test_set = [e for e in train_set if e["pub_year"] == 2018]
print(f"Test set len :{len(test_set)}")

test_set = [e for e in test_set if e["journal_nlmid"] in selectively_indexed_journals]
print(f"Filtered test set len :{len(test_set)}")

print(f"BmCS processed :{len([e for e in test_set if e['bmcs_processed_date'] is not None ])}")
print(f"Date completed null :{len([e for e in test_set if e['date_completed'] is None ])}")
print(f"Is indexed :{len([e for e in test_set if e['is_indexed'] ])}")

test_set = random.sample(test_set, len(test_set))

with gzip.open(test_set_file, "wt", encoding="utf8") as save_file:
    json.dump(test_set, save_file, ensure_ascii=False, indent=4)
