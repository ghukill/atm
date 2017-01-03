# api.py
# -*- coding: utf-8 -*-


from collections import defaultdict
import json
import logging
logging.basicConfig(level=logging.DEBUG)
import md5
import os
import string

# Dropbox
import dropbox

# GenSim
from gensim import corpora, models, similarities

# LevelDB
import leveldb

# NLTK
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer

# Slate
import slate

# localConfig
import localConfig



# initialize LevelDB instance
db = leveldb.LevelDB('./db')


# Dropbox Client
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
					a.load_local(entry.name.encode('utf-8'))
					try:
						a.extract_text()
					except:
						logging.warning('could not exract text, continuing')
				if limit is not None:
					count += 1
					if count >= limit:
						return


	def download_file(self, filename, extract_text=False):
		'''
		download specific file
		todo: refactor the download part shared with above
		# problem: Borglund_Nuldén_2012.pdf
		'''
		entry = self.api.files_get_metadata('%s/%s' % (self.dropbox_path, filename))
		r = self.api.files_download_to_file('%s/%s' % (self.corpora_path, entry.name), entry.path_lower)
		if extract_text:
			a = Article()
			a.load_local(entry.name.encode('utf-8'))
			try:
				a.extract_text()
			except:
				logging.warning('could not exract text, continuing')


	def list_files(self):
		'''
		lists DROPBOX_PATH
		'''
		file_list = []
		for entry in self.api.files_list_folder(localConfig.DROPBOX_PATH).entries:
			if type(entry) == dropbox.files.FileMetadata:
				file_list.append(entry)
		return file_list



# Article class
class Article(object):


	def __init__(self):
		self.corpora_path = localConfig.CORPORA_PATH
		self.stoplist = set(stopwords.words('english'))
		self.filename = False
		self.id = False
		self.file_handle = False
		self.raw_text = False
		self.tokens = None


	def gen_id(self, filename):
		'''
		hashes filename with md5, return hexdigest
		'''
		id = md5.new(filename).hexdigest()
		logging.debug("generated id: %s" % id)
		return id


	def load_local(self, filename):
		'''
		opens filename that combines the corpora path and provided filename
		'''
		if os.path.exists('%s/%s' % (self.corpora_path, filename)):
			logging.debug('file found')
			self.filename = filename
			# set id
			self.id = self.gen_id(self.filename)
			self.file_handle = open('%s/%s' % (self.corpora_path, filename))
			# check for extracted tokens in db
			try:
				self.tokens = json.loads(db.Get(self.id))
				logging.debug('tokens found in db')
			except KeyError:
				logging.debug('could not find tokens in db')
			except UnicodeEncodeError:
				logging.debug('Unicode error...')	
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
			db.Put(self.id, json.dumps(self.tokens))


# Model class
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


	def load_dict_and_corpora(self,filename):
		'''
		see above for selecting other corpora serializations
		'''
		# load corpora
		target_path = '%s/%s.mm' % (localConfig.INDEX_PATH, filename)
		if os.path.exists(target_path):
			logging.debug("loading serialized corpora model: %s.mm" % filename)
			self.corpus = corpora.MmCorpus(target_path)
		# load dictionary
		target_path = '%s/%s.dict' % (localConfig.INDEX_PATH, filename)
		if os.path.exists(target_path):
			logging.debug("loading serialized dictionary: %s.dict" % filename)
			self.dictionary = corpora.Dictionary.load(target_path)


	# def similarity(self):
	# 	self.lsi = models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=2)




class Tests(object):


	plain_filename = 'Anderson_2008.pdf'
	unicode_filename = 'Borglund_Nuldén_2012.pdf'


	def download_unicode_filename(self):
		d = DBXClient()
		r = d.download_file(self.unicode_filename, extract_text=True)


	def load_local_unicode_filename(self):
		a = Article()
		a.load_local(self.unicode_filename)


	def download_plain_filename(self):
		d = DBXClient()
		r = d.download_file(self.plain_filename, extract_text=True)


	def load_local_plain_filename(self):
		a = Article()
		a.load_local(self.plain_filename)
