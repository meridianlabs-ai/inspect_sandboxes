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
| src/inspect\_sandboxes/daytona/\_compose.py    |       67 |        4 |       44 |        6 |     91% |138, 143, 147->158, 148->147, 152, 162 |
| src/inspect\_sandboxes/daytona/\_daytona.py    |      246 |       29 |       60 |        9 |     86% |80, 90, 111, 162, 169-171, 188, 199, 258-262, 303, 310, 338, 344-348, 360->363, 392-393, 417, 457-458, 467-468, 483-486 |
| src/inspect\_sandboxes/modal/\_\_init\_\_.py   |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/modal/\_compose.py      |      103 |        5 |       64 |        4 |     95% |26->29, 177-178, 185-186, 228->233, 229->228, 234 |
| src/inspect\_sandboxes/modal/\_modal.py        |      260 |       57 |       64 |        6 |     77% |75, 85, 234-239, 300-313, 318-319, 342, 351-355, 368, 372-373, 396-399, 406-407, 420, 425, 430-433, 454-457, 460-462, 465-480, 484 |
| **TOTAL**                                      |  **721** |  **110** |  **244** |   **26** | **84%** |           |


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