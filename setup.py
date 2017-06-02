from setuptools import setup
from setuptools import find_packages

install_requires = [
        'numpy',
        'theano',
        'lasagne',
        ]

setup(
      name="simpleQnet",
      version="0.1.0",
      description="Simple Q network to work with openAI environments like atari games.",
      author="Toni Gabas",
      author_email="a.gabas@aist.go.jp",
      url="",
      long_description=
      """
        Trainable network and environment handlers to learn from openAI's environments.
      """,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      )

