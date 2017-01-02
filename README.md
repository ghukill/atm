# Article Topic Modeling (atm)

## Helpful resources
  * http://stats.stackexchange.com/a/115150
  * https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
  * https://radimrehurek.com/gensim/tut1.html
  * http://journalofdigitalhumanities.org/2-1/topic-modeling-and-digital-humanities-by-david-m-blei/
  * https://github.com/ariddell/lda/

## Prep NLTK

Download some of the corpora required by NLTK

```
import nltk
nltk.download() # opens pop-up, download corpora/stopwords.txt
```

## Discussion

This is currently using flat storage for downloaded articles, indexes, etc., which is fine.  But if we were to leverage a DB, it would make possible things like extracting raw text and tokenizing, then saving that output alongside the original PDF.  We could do that now with flat files, but it requires falling back on naming conventions.
