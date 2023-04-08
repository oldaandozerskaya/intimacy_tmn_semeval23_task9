The repo contains the notebook and generated texts created for the <a href='https://sites.google.com/umich.edu/semeval-2023-tweet-intimacy/home'>
SemEval-2023 Task 9: Multilingual Tweet Intimacy Analysis</a> Shared Task.
The corresponding paper is: <i>Anna Glazkova. 2023. tmn at SemEval-2023 Task 9: Multilingual Tweet Intimacy Detection using XLM-T, Google Translate, and Ensemble Learning. Proceedings of SemEval-2023</i> (in print). 

<h2>Table of Contents</h2>

<ul>
 <li><a href = 'https://github.com/oldaandozerskaya/intimacy_tmn_semeval23_task9/blob/main/tmn_intimacy.py'>The code for fine-tuning XLM-T</a></li>
 <li>Additional data obtained with Google Translator: trainslated_train.pickle, translated_test.pickle</a></li>
 
 These files can be read by:
```
import pickle
with open('translated_train.pickle', 'rb') as f:
  translated_texts = pickle.load(f)
```
</ul>



