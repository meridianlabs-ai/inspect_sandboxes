# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/meridianlabs-ai/inspect_sandboxes/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                               |    Stmts |     Miss |   Branch |   BrPart |  Cover |   Missing |
|--------------------------------------------------- | -------: | -------: | -------: | -------: | -----: | --------: |
| src/inspect\_sandboxes/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |   100% |           |
| src/inspect\_sandboxes/\_registry.py               |        3 |        3 |        0 |        0 |     0% |       2-4 |
| src/inspect\_sandboxes/\_util/\_\_init\_\_.py      |        0 |        0 |        0 |        0 |   100% |           |
| src/inspect\_sandboxes/\_util/compose.py           |       72 |       72 |       30 |        0 |     0% |     1-172 |
| src/inspect\_sandboxes/\_util/dind\_compose.py     |       61 |       61 |       28 |        0 |     0% |     8-120 |
| src/inspect\_sandboxes/\_util/naming.py            |       42 |       42 |       14 |        0 |     0% |      3-72 |
| src/inspect\_sandboxes/daytona/\_\_init\_\_.py     |        0 |        0 |        0 |        0 |   100% |           |
| src/inspect\_sandboxes/daytona/\_compose.py        |      117 |      117 |       68 |        0 |     0% |     1-319 |
| src/inspect\_sandboxes/daytona/\_daytona.py        |      183 |      183 |       48 |        0 |     0% |     3-356 |
| src/inspect\_sandboxes/daytona/\_dind\_env.py      |      184 |      184 |       50 |        0 |     0% |     3-444 |
| src/inspect\_sandboxes/daytona/\_dind\_project.py  |      178 |      178 |       40 |        0 |     0% |     3-454 |
| src/inspect\_sandboxes/daytona/\_exec\_capture.py  |       78 |       78 |        8 |        0 |     0% |    13-250 |
| src/inspect\_sandboxes/daytona/\_retry.py          |       26 |       26 |        6 |        0 |     0% |     13-81 |
| src/inspect\_sandboxes/daytona/\_sandbox\_utils.py |      127 |      127 |       32 |        0 |     0% |     3-290 |
| src/inspect\_sandboxes/daytona/\_single\_env.py    |      151 |      151 |       26 |        0 |     0% |     3-327 |
| src/inspect\_sandboxes/e2b/\_\_init\_\_.py         |        0 |        0 |        0 |        0 |   100% |           |
| src/inspect\_sandboxes/e2b/\_compose.py            |      105 |      105 |       54 |        0 |     0% |     1-239 |
| src/inspect\_sandboxes/e2b/\_dind\_env.py          |      171 |      171 |       46 |        0 |     0% |     3-382 |
| src/inspect\_sandboxes/e2b/\_dind\_project.py      |      159 |      159 |       38 |        0 |     0% |    18-430 |
| src/inspect\_sandboxes/e2b/\_e2b.py                |      240 |      240 |       90 |        0 |     0% |     3-442 |
| src/inspect\_sandboxes/e2b/\_retry.py              |       32 |       32 |        8 |        0 |     0% |    15-103 |
| src/inspect\_sandboxes/e2b/\_single\_env.py        |      137 |      137 |       22 |        0 |     0% |     3-305 |
| src/inspect\_sandboxes/e2b/\_template.py           |       37 |       37 |        2 |        0 |     0% |    24-121 |
| src/inspect\_sandboxes/modal/\_\_init\_\_.py       |        0 |        0 |        0 |        0 |   100% |           |
| src/inspect\_sandboxes/modal/\_compose.py          |      182 |      182 |      114 |        0 |     0% |     1-471 |
| src/inspect\_sandboxes/modal/\_modal.py            |      345 |      345 |       98 |        0 |     0% |     1-832 |
| **TOTAL**                                          | **2630** | **2630** |  **822** |    **0** | **0%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/meridianlabs-ai/inspect_sandboxes/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/meridianlabs-ai/inspect_sandboxes/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/meridianlabs-ai/inspect_sandboxes/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/meridianlabs-ai/inspect_sandboxes/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmeridianlabs-ai%2Finspect_sandboxes%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/meridianlabs-ai/inspect_sandboxes/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.