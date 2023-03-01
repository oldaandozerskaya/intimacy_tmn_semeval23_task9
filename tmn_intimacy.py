#Read data
"""

import pickle
import pandas as pd

#read train
df = pd.read_csv("train.csv")
df = df.sample(frac=1, random_state = 0)
train_texts = list(df.text.values)
train_labels = list(df.label.values)

#read test
df = pd.read_csv("test.csv")
test_texts = list(df.text.values)

#read translated train
with open('translated_train.pickle', 'rb') as f:
    tr_train_texts = pickle.load(f)

#read translated test
with open('test.pickle', 'rb') as f:
    tr_test_texts = pickle.load(f)

#putting it all together
train_texts = [train_texts[i] + ' </s></s> ' + tr_train_texts[i] for i in range(len(train_texts))]
test_texts = [test_texts[i] + ' </s></s> ' + tr_test_texts[i] for i in range(len(test_texts))]

"""#XLM"""

!pip install simpletransformers
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

#fine-tuning
import numpy as np

train_df = pd.DataFrame({"text": train_texts, "labels": train_labels})
model_args = ClassificationArgs(num_train_epochs=3, overwrite_output_dir=True, \
                            no_save = True, max_seq_length=128, regression = True)
model = ClassificationModel(
  #"xlmroberta", "cardiffnlp/twitter-xlm-roberta-base", args=model_args, use_cuda=True, num_labels=1
  #"xlmroberta", "xlm-roberta-base", args=model_args, use_cuda=True, num_labels=1
  "bert", "bert-base-multilingual-cased", args=model_args, use_cuda=True, num_labels=1
)

model.train_model(train_df)

predictions, raw_output = model.predict(list(test_texts))
print(mean_squared_error(predictions, test_labels))
print(r2_score(predictions, test_labels) ** (0.5))