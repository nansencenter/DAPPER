Contributing
==========================

Making a release
--------------------------

- ``cd DAPPER``
- Bump version number in ``__init__.py``
- Merge dev1 into master::

    git checkout master
    git merge --no-commit --no-ff dev1
    # Fix conflicts, e.g
    # git rm <unwanted-file>
    git commit

- Make docs (including bib)
- Tag::
  
    git tag -a v$(python setup.py --version) -m 'My description'
    git push origin --tags

- Clean::
  
    rm -rf build/ dist *.egg-info .eggs

- Add new files to ``package_data`` and ``packages`` in ``setup.py``

- Build::

    ./setup.py sdist bdist_wheel

- Upload to PyPI::

    twine upload --repository pypi dist/*


- Upload to Test.PyPI::

    twine upload --repository testpypi dist/*

  where ``~/.pypirc`` contains::

    [distutils]
    index-servers=
                    pypi
                    testpypi

    [pypi]
    username: myuser
    password: mypass

    [testpypi]
    repository: https://test.pypi.org/legacy/
    username: myuser
    password: mypass

- Upload to Test.PyPI::

    git checkout dev1




Test installation
--------------------------

- Install from Test.PyPI::
  
    pip install --extra-index-url https://test.pypi.org/simple/ DA-DAPPER

- Install from PyPI::
  
    pip install DA-DAPPER

  - Install into specific dir (includes all of the dependencies)::
    
      pip install DA-DAPPER -t MyDir

  - Install with options::
  
      pip install DA-DAPPER[Qt,MP]

- Install from local (makes installation accessible from everywhere)::
  
    pip install -e .
