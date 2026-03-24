# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/meridianlabs-ai/inspect_sandboxes/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|----------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/inspect\_sandboxes/\_\_init\_\_.py         |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/\_registry.py           |        2 |        2 |        0 |        0 |      0% |       2-3 |
| src/inspect\_sandboxes/\_util/\_\_init\_\_.py  |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/\_util/compose.py       |       30 |        0 |       12 |        1 |     98% |    22->21 |
| src/inspect\_sandboxes/\_version.py            |       13 |       13 |        0 |        0 |      0% |      4-34 |
| src/inspect\_sandboxes/daytona/\_\_init\_\_.py |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/daytona/\_compose.py    |       69 |        4 |       46 |        6 |     91% |143, 148, 152->163, 153->152, 157, 167 |
| src/inspect\_sandboxes/daytona/\_daytona.py    |      274 |       28 |       66 |        8 |     88% |106, 116, 137, 188, 195-197, 214, 227-228, 241->260, 254-256, 310-314, 355, 395, 417->420, 449-450, 474, 514-515, 524-525, 540-543 |
| src/inspect\_sandboxes/modal/\_\_init\_\_.py   |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/modal/\_compose.py      |      119 |        5 |       72 |        3 |     96% |212-213, 220-221, 263->268, 264->263, 269 |
| src/inspect\_sandboxes/modal/\_modal.py        |      270 |       54 |       64 |        6 |     79% |98, 108, 260-265, 327-340, 369, 379-382, 395, 399-400, 422-425, 432-433, 446, 453, 458-461, 487-490, 494-496, 500-515, 519 |
| **TOTAL**                                      |  **777** |  **106** |  **260** |   **24** | **86%** |           |


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