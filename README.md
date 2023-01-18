# Comment classifier for retail
Based in Spanish comments for a tech business

This is my first machine learning project. It utilizes a dataset from labeled comments to accurately classify the main subject a comment is talking about (eg. good service, kindness, efficency).

## word_bank.py
Creates 1-gram, 2-gram and 3-gram dictionaries with its corresponding probabilities, based on a big text document (in Spanish).

## spell_fix.py
Function, heavily inspired in Norvig's Spelling Corrector, with the added feature of n-gram search.

This function it's not vectorized yet, so it needs to be applied to each row at a time, causing the most time consumption whenever applied.

## model_selection.py

Different models are tried out, results (measured by accuracy) as shown

![alt text](https://github.com/irving-m/comment-clasiffier-retail/blob/master/results.JPG?raw=true)

Results indicate linear support vector machine yields the best resulsts for this task, across all the possible data labels.

## svc_classifier.py

Finally, this last bit applies the found model (saved as linearsvc.pickle), to any new dataset.
