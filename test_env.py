import pytest

def test_jaxlib_version():
  from importlib.metadata import version
  assert "cuda" in version('jaxlib')

def test_import_PIL():
  from PIL import Image

def test_import_tf():
  import tensorflow as tf

def test_import_keras():
  from tensorflow import keras