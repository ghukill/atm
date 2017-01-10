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

## Article Similarity 
```
# import saved corpus, model, and index
m = atm.Model('MODEL_NAME')
m.load_corpora()
m.load_lda()
# populate article_hash (Move to other approach...)
m.get_all_articles()
# generate similarity index (or load)
m.gen_similarity_index()
# open article
a = atm.Article()
a.load_local('ARTICLE_FILENAME')
# generate attribute as vector bag of words (vec_bow)
a.as_vec_bow()
# run article similarity query, combing article and model
# this saves to `article.sims`
m.article_similarity_query(a)
# view similarities
print a.sims
# document index with percentages returned, to get filename from doc index
doc_filename = m.get_article_filename(DOC_INDEX_FROM_SIMS)
# OR, view filenames of sims, with optional results parameter
a.sims_as_filenames(m, results=10)

```

## Discussion

This is currently using flat storage for downloaded articles, indexes, etc., which is fine.  But if we were to leverage a DB, it would make possible things like extracting raw text and tokenizing, then saving that output alongside the original PDF.  We could do that now with flat files, but it requires falling back on naming conventions.
