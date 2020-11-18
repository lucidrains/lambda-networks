from setuptools import setup, find_packages

setup(
  name = 'lambda-networks',
  packages = find_packages(),
  version = '0.4.0',
  license='MIT',
  description = 'Lambda Networks - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/lambda-networks',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'image recognition'
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.3'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)