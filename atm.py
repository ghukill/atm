# api.py
# -*- coding: utf-8 -*-

import slate

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

import logging
logging.basicConfig(level=logging.DEBUG)

import gensim
from gensim import corpora

import os
from collections import defaultdict

import dropbox

import localConfig

import leveldb
import json


# initialize db
db = leveldb.LevelDB('./db')


class DBXClient(object):

	def __init__(self):
		self.dropbox_path = localConfig.DROPBOX_PATH
		self.corpora_path = localConfig.CORPORA_PATH
		self.api = dropbox.Dropbox(localConfig.DROPBOX_ACCESS_TOKEN)


	def download_files(self, limit=None, extract_text=False):
		'''
		lists DROPBOX_PATH, downloads files
		'''
		count = 0
		if limit is not None:
			logging.info("limiting downloads to: %d" % limit)
		for entry in self.api.files_list_folder(localConfig.DROPBOX_PATH).entries:
			logging.debug("working on: %s" % entry.name)
			if type(entry) == dropbox.files.FileMetadata:
				r = self.api.files_download_to_file('%s/%s' % (self.corpora_path, entry.name), entry.path_lower)
				logging.debug(r)
				# if extract text and saving to db, do that now
				if extract_text:
					a = Article()
					a.load_local(entry.name)
					try:
						a.extract_text()
					except:
						logging.warning('could not exract text, continuing')
				if limit is not None:
					count += 1
					if count >= limit:
						return


	def list_files(self):
		'''
		lists DROPBOX_PATH
		'''
		file_list = []
		for entry in self.api.files_list_folder(localConfig.DROPBOX_PATH).entries:
			if type(entry) == dropbox.files.FileMetadata:
				file_list.append(entry)
		return file_list



class Article(object):


	def __init__(self):
		self.corpora_path = localConfig.CORPORA_PATH
		self.stoplist = set(stopwords.words('english'))
		self.filename = False
		self.file_handle = False
		self.raw_text = False
		self.tokens = None


	def load_local(self,filename):
		'''
		opens filename that combines the corpora path and provided filename
		'''
		if os.path.exists('%s/%s' % (self.corpora_path, filename)):
			logging.debug('file found')
			self.filename = filename
			self.file_handle = open('%s/%s' % (self.corpora_path, filename))
			# check for extracted tokens in db
			try:
				logging.debug('tokens found in db')
				self.tokens = json.loads(db.Get(self.filename))
			except KeyError:
				logging.debug('could not find tokens in db')
		else:
			logging.debug('file not found')
			return False


	def extract_text(self):
		'''
		requires open self.file_handle
		'''
		if self.file_handle:
			# open slate document
			logging.debug('opening as slate document')
			self.slate_doc = slate.PDF(self.file_handle)
			logging.debug('decoding as utf-8')
			# decode as utf-8
			self.raw_text = "\n".join(self.slate_doc).decode('utf-8')
			logging.debug('extracting words, removing stopwords and punctuation')
			# remove stopwords
			self.tokens = [word for word in self.raw_text.lower().split() if word not in self.stoplist]
			# remove words that only appear once
			logging.debug('removing words that appear > than WORD_COUNT_MIN: %d' % localConfig.WORD_COUNT_MIN)
			frequency = defaultdict(int)
			for token in self.tokens:
				frequency[token] += 1
			self.tokens = [token for token in self.tokens if frequency[token] > localConfig.WORD_COUNT_MIN]
			# save raw text to level db
			db.Put(self.filename, json.dumps(self.tokens))


class Model(object):

	'''
	This is where multiple articles are aggregated as a gensim model
	'''

	def __init__(self, name):
		self.texts = []
		self.article_hash = {}
		self.failed = []
		self.name = name


	def get_all_articles(self):

		for filename in os.listdir(localConfig.CORPORA_PATH):
			try:
				logging.debug("\n\n")
				logging.debug("including article %s" % filename)
				# using Article class
				a = Article()
				a.load_local(filename)
				# if tokens have not been extracted and save to db, do that now
				if a.tokens == None:
					logging.debug("tokens not found in db, extracting now")
					a.extract_text()
				self.texts.append(a.tokens)
				self.article_hash[filename] = len(self.texts) - 1
			except:
				logging.warning("failed on %s" % filename)
				self.failed.append(filename)


	def gen_corpora(self):
		logging.debug("creating corpora dictionary for texts: %s.dict" % self.name)
		# creating gensim dictionary
		self.dictionary = corpora.Dictionary(self.texts)
		self.dictionary.save('%s/%s.dict' % (localConfig.INDEX_PATH, self.name))
		# creating gensim corpus
		self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
		'''
		Consider future options for alternate formats:
			Other formats include Joachim’s SVMlight format, Blei’s LDA-C format and GibbsLDA++ format.

			>>> corpora.SvmLightCorpus.serialize('/tmp/corpus.svmlight', corpus)
			>>> corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
			>>> corpora.LowCorpus.serialize('/tmp/corpus.low', corpus)
		'''
		corpora.MmCorpus.serialize('%s/%s.mm' % (localConfig.INDEX_PATH, self.name), self.corpus)
		logging.debug('finis.')


	def load_corpora(self,filename):
		'''
		see above for selecting other corpora serializations
		'''
		target_path = '%s/%s.mm' % (localConfig.INDEX_PATH, filename)
		if os.path.exists(target_path):
			logging.debug("loading serialized corpora model: %s.mm" % filename)
			self.corpus = corpora.MmCorpus(target_path)




