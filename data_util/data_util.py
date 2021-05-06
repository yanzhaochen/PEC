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

    conversations = []
    conversation = []
    with open(txt_file, "rt") as f:
        for us_id, line in enumerate(f):
            # print(us_id, line)
            # 0 'HeWentToJared91 ---+--- found out this morning i got a job promotion ! ! !\n'
            if line[0] != "*":
                line = line.strip()

                fields = line.split(" ---+--- ")
                if len(fields) == 1:
                    print(us_id, fields[0])
                if len(fields) == 2:
                    # print(us_id, "****" + fields[0] + "****", fields[1])
                    # 0 ****HeWentToJared91**** found out this morning i got a job promotion ! ! !

                    sent = [fields[1].replace("'", "")]
                    # 'found out this morning i got a job promotion ! ! !'

                    labeldict = goemotions(sent)
                    # pprint(labeldict)
                    # [{'labels': ['joy', 'neutral'], 'scores': [0.37095714, 0.5489171]}]

                    labels = labeldict[0]["labels"]
                    # print(labels)
                    # ['joy', 'neutral']

                    uttr = (fields[0], labels + sent)
                    # print(uttr)
                    # ('HeWentToJared91', ['joy', 'neutral', 'found out this morning i got a job promotion ! ! !'])
                    conversation.append(uttr)

            # end of conversation
            elif line[0] == "*":
                conversations.append(conversation)
                conversation = []

    with open(pkl_file, "wb",) as ff:
        pickle.dump(conversations, ff)
    print("finished")


if __name__ == "__main__":
    source = "data/cleaned/happy/train.txt"
    target = "data/cleaned/happy/train.pkl"
    txt_to_pkl(source, target)
