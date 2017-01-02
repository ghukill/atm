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

import dropbox

import localConfig


class DBXClient(object):

	def __init__(self):
		self.dropbox_path = localConfig.DROPBOX_PATH
		self.corpora_path = localConfig.CORPORA_PATH
		self.api = dropbox.Dropbox(localConfig.DROPBOX_ACCESS_TOKEN)


	def download_files(self, limit=None):
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
		self.file_handle = False


	def load_local(self,filename):
		if os.path.exists('%s/%s' % (self.corpora_path, filename)):
			logging.debug('file found')
			self.file_handle = open('%s/%s' % (self.corpora_path, filename))


	def extract_text(self):
		'''
		requires open self.file_handle
		'''
		if self.file_handle:
			self.slate_doc = slate.PDF(self.file_handle)
			self.raw_text = "\n".join(self.slate_doc).decode('utf-8')
			


