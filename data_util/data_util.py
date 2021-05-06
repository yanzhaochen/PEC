import pickle
from pprint import pprint

from transformers import BertTokenizer

from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline


def txt_to_pkl(txt_file, pkl_file):
    sentitokenizer = BertTokenizer.from_pretrained(
        "monologg/bert-base-cased-goemotions-ekman"
    )
    sentimodel = BertForMultiLabelClassification.from_pretrained(
        "monologg/bert-base-cased-goemotions-ekman"
    )

    goemotions = MultiLabelPipeline(
        model=sentimodel, tokenizer=sentitokenizer, threshold=0.3
    )

    utters = []
    utter = []
    with open(txt_file, "rt") as f:
        for us_id, line in enumerate(f):
            print(us_id, line)
            if line[0] != "*":
                line = line.strip()
                fields = line.split(" ---+--- ")
                if len(fields) == 1:
                    print(us_id, fields[0])
                if len(fields) == 2:
                    print(us_id, "****" + fields[0] + "****", fields[1])
                    sents = [fields[1].replace("'", "")]
                    labeldict = goemotions(sents)
                    pprint(labeldict)
                    labels = labeldict[0]["labels"]
                    print(labels)
                    t = (fields[0], labels + sents)
                    print(t)
                    utter.append(t)
            elif line[0] == "*":
                utters.append(utter)
                utter = []

    print(utters)
    with open(pkl_file, "wb",) as ff:
        pickle.dump(utters, ff)
    print("finished")


if __name__ == "__main__":
    source = "data/cleaned/happy/train.txt"
    target = "data/cleaned/happy/train.pkl"
    txt_to_pkl(source, target)
