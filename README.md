# Comment classifier for retail
Based in Spanish comments for a tech business

This is my first machine learning project. I might still need to tidy up my files because I made this before knowing how to use git and github.

## word_bank.py
Creates 1-gram, 2-gram and 3-gram dictionaries with its corresponding probabilities, based on a big text document (in Spanish).

## spell_fix.py
Function, heavily inspired in Norvig's Spelling Corrector, with the added feature of n-gram search.

This function it's not vectorized yet, so it needs to be applied to each row at a time, causing the most time consumption.

## model_selection.py

Different models are tried out, results as shown

![alt text](https://github.com/irving-m/comment-clasiffier-retail/blob/master/results.JPG?raw=true)

