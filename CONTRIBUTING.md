# Contributing

**We appreciate all kinds of help, so thank you!**

First please read the overall project contributing guidelines. These are
included in the Qiskit documentation here:

https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md

## Contributing to Qiskit Machine Learning

In addition to the general guidelines above there are specific details for
contributing to Qiskit Machine Learning.

You should first install the python development libraries by running
`pip install -r requirements-dev.txt` from the root of the
Machine Learning repository clone and then
follow the  guidelines below.

### Project Code Style.

Code in Qiskit Machine Learning should conform to PEP8 and style/lint checks are run to validate
this.  Line length must be limited to no more than 100 characters. Docstrings
should be written using the Google docstring format.

### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the _code style_ of this project and successfully
   passes the _unit tests_. Machine Learning uses [Pylint](https://www.pylint.org) and
   [PEP8](https://www.python.org/dev/peps/pep-0008) style guidelines.
   
   You can run
   ```shell script
   make lint
   make style 
   ```
   from the root of the Machine Learning repository clone for lint and style conformance checks.

   If your code fails the local style checks (specifically the black
   code formatting check) you can use `make black` to automatically
   fix update the code formatting.
   
   For unit testing please see [Testing](#testing) section below.
   
2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* accordingly.
   
   The documentation will be built/tested using Sphinx and should be free
   from errors and warnings.
   You will need to [install pandoc](https://pandoc.org/installing.html) first.
   
   Then you can run
   ```shell script
    make html
   ```
   from the root of the Machine Learning repository clone. You might also like to check the html output
   to see the changes formatted output is as expected. You will find an index.html
   file in docs\_build\html and you can navigate from there.
   
   Please note that a spell check is run in CI, on the docstrings.
   
   You can run `make spell` locally to check spelling though you would need to
   [install pyenchant](https://pyenchant.github.io/pyenchant/install.html) and be using
   hunspell-en-us as is used by the CI. 
   
   For some words, such as names, technical terms, referring to parameters of the method etc., 
   that are not in the en-us dictionary and get flagged as being misspelled, despite being correct,
   there is a [.pylintdict](./.pylintdict) custom word list file, in the root of the Machine Learning repo,
   where such words can be added, in alphabetic order, as needed.
   
3. If it makes sense for your change that you have added new tests that
   cover the changes and any new function.
   
4. Ensure that if your change has an end user facing impact (new feature,
   deprecation, removal etc) that you have added a reno release note for
   that change and that the PR is tagged for the changelog.

5. Ensure all code, including unit tests, has the copyright header. The copyright
   date will be checked by CI build. The format of the date(s) is _year of creation,
   last year changed_. So for example:
   
   > \# (C) Copyright IBM 2018, 2021.

   If the _year of creation_ is the same as _last year changed_ then only
   one date is needed, for example:

   > \# (C) Copyright IBM 2021.
                                                                                                                                                                                                 
   If code is changed in a file make sure the copyright includes the current year.
   If there is just one date and it's a prior year then add the current year as the 2nd date, 
   otherwise simply change the 2nd date to the current year. The _year of creation_ date is
   never changed.

### Changelog generation

A changelog is manually generated as part of the release process from the release notes.

### Release Notes

When making any end user facing changes in a contribution we have to make sure
we document that when we release a new version of qiskit-machine-learning. The expectation
is that if your code contribution has user facing changes that you will write
the release documentation for these changes. This documentation must explain
what was changed, why it was changed, and how users can either use or adapt
to the change. The idea behind release documentation is that when a naive
user with limited internal knowledge of the project is upgrading from the
previous release to the new one, they should be able to read the release notes,
understand if they need to update their program which uses qiskit, and how they
would go about doing that. It ideally should explain why they need to make
this change too, to provide the necessary context.

To make sure we don't forget a release note or if the details of user facing
changes over a release cycle we require that all user facing changes include
documentation at the same time as the code. To accomplish this we use the
[reno](https://docs.openstack.org/reno/latest/) tool which enables a git based
workflow for writing and compiling release notes.

#### Adding a new release note

Making a new release note is quite straightforward. Ensure that you have reno
installed with::

    pip install -U reno

Once you have reno installed you can make a new release note by running in
your local repository checkout's root::

    reno new short-description-string

where short-description-string is a brief string (with no spaces) that describes
what's in the release note. This will become the prefix for the release note
file. Once that is run it will create a new yaml file in releasenotes/notes.
Then open that yaml file in a text editor and write the release note. The basic
structure of a release note is restructured text in yaml lists under category
keys. You add individual items under each category and they will be grouped
automatically by release when the release notes are compiled. A single file
can have as many entries in it as needed, but to avoid potential conflicts
you'll want to create a new file for each pull request that has user facing
changes. When you open the newly created file it will be a full template of
the different categories with a description of a category as a single entry
in each category. You'll want to delete all the sections you aren't using and
update the contents for those you are. For example, the end result should
look something like::

```yaml
features:
  - |
    Introduced a new feature foo, that adds support for doing something to
    ``QuantumCircuit`` objects. It can be used by using the foo function,
    for example::

      from qiskit import foo
      from qiskit import QuantumCircuit
      foo(QuantumCircuit())

  - |
    The ``qiskit.QuantumCircuit`` module has a new method ``foo()``. This is
    the equivalent of calling the ``qiskit.foo()`` to do something to your
    QuantumCircuit. This is the equivalent of running ``qiskit.foo()`` on
    your circuit, but provides the convenience of running it natively on
    an object. For example::

      from qiskit import QuantumCircuit

      circ = QuantumCircuit()
      circ.foo()

deprecations:
  - |
    The ``qiskit.bar`` module has been deprecated and will be removed in a
    future release. Its sole function, ``foobar()`` has been superseded by the
    ``qiskit.foo()`` function which provides similar functionality but with
    more accurate results and better performance. You should update your calls
    ``qiskit.bar.foobar()`` calls to ``qiskit.foo()``.
```

You can also look at other release notes for other examples.

You can use any restructured text feature in them (code sections, tables,
enumerated lists, bulleted list, etc.) to express what is being changed as
needed. In general, you want the release notes to include as much detail as
needed so that users will understand what has changed, why it changed, and how
they'll have to update their code.

After you've finished writing your release notes you'll want to add the note
file to your commit with `git add` and commit them to your PR branch to make
sure they're included with the code in your PR.

#### Generating the release notes

After release notes have been added if you want to see what the full output of
the release notes. In general the output from reno that we'll get is a rst
(ReStructuredText) file that can be compiled by
[sphinx](https://www.sphinx-doc.org/en/master/). To generate the rst file you
use the ``reno report`` command. If you want to generate the full Machine Learning release
notes for all releases (since we started using reno during 0.9) you just run::

    reno report

but you can also use the ``--version`` argument to view a single release (after
it has been tagged::

    reno report --version 0.5.0

At release time ``reno report`` is used to generate the release notes for the
release and the output will be submitted as a pull request to the documentation
repository's [release notes file](
https://github.com/Qiskit/qiskit/blob/main/docs/release_notes.rst)

#### Building release notes locally

Building The release notes are part of the standard qiskit-machine-learning documentation
builds. To check what the rendered html output of the release notes will look
like for the current state of the repo you can run: `tox -edocs` which will
build all the documentation into `docs/_build/html` and the release notes in
particular will be located at `docs/_build/html/release_notes.html`

## Installing Qiskit Machine Learning from source

Please see the [Installing Qiskit Machine Learning from
Source](https://github.com/qiskit-community/qiskit-machine-learning#installation)
section of the Qiskit documentation.

Note: Machine Learning depends on Qiskit, and has an optional dependence on Aer, so
should be installed too.

Machine Learning also has some other optional dependents see 
[Machine Learning optional installs](https://github.com/qiskit-community/qiskit-machine-learning#optional-installs) for
further information. Unit tests that require any of the optional dependents will check
and skip the test if not installed.

### Testing

Once you've made a code change, it is important to verify that your change
does not break any existing tests and that any new tests that you've added
also run successfully. Before you open a new pull request for your change,
you'll want to run the test suite locally.

The test suite can be run from a command line or via your IDE. You can run `make test` which will
run all unit tests. Another way to run the test suite is to use
[**tox**](https://tox.readthedocs.io/en/latest/#). For more information about using tox please
refer to
[Qiskit CONTRIBUTING](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md#test)
Test section. However please note Machine Learning does not have any
[online tests](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md#online-tests)
nor does it have
[test skip
 options](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md#test-skip-options).    

### Development Cycle

The development cycle for qiskit-machine-learning is informed by release plans in the 
[Qiskit rfcs repository](https://github.com/Qiskit/rfcs)
 
### Branches

* `main`:

The main branch is used for development of the next version of qiskit-machine-learning.
It will be updated frequently and should not be considered stable. The API
can and will change on main as we introduce and refine new features.

* `stable/*`:
The stable branches are used to maintain the most recent released versions of
qiskit-machine-learning. It contains the versions of the code corresponding to the minor
version release in the branch name release for The API on these branches are
stable and the only changes merged to it are bugfixes.

### Release Cycle

From time to time, we will release brand-new versions of Qiskit Machine Learning. These
are well-tested versions of the software.

The `stable/*` branches should only receive changes in the form of bug
fixes.

## Dealing with the git blame ignore list

In the qiskit-machine-learning repository we maintain a list of commits for git blame
to ignore. This is mostly commits that are code style changes that don't
change the functionality but just change the code formatting (for example,
when we migrated to use black for code formatting). This file,
`.git-blame-ignore-revs` just contains a list of commit SHA1s you can tell git
to ignore when using the `git blame` command. This can be done one time
with something like

```
git blame --ignore-revs-file .git-blame-ignore-revs qiskit_machine_learning/version.py
```

from the root of the repository. If you'd like to enable this by default you
can update your local repository's configuration with:

```
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

which will update your local repositories configuration to use the ignore list
by default.
