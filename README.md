# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/meridianlabs-ai/inspect_sandboxes/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/inspect\_sandboxes/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/\_registry.py               |        2 |        2 |        0 |        0 |      0% |       2-3 |
| src/inspect\_sandboxes/\_util/\_\_init\_\_.py      |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/\_util/compose.py           |       30 |        0 |       12 |        1 |     98% |   22-\>21 |
| src/inspect\_sandboxes/\_util/naming.py            |       24 |        0 |        8 |        0 |    100% |           |
| src/inspect\_sandboxes/\_version.py                |       11 |       11 |        0 |        0 |      0% |      3-24 |
| src/inspect\_sandboxes/daytona/\_\_init\_\_.py     |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/daytona/\_compose.py        |      108 |        6 |       64 |        8 |     92% |113-\>112, 115-116, 251, 256, 260-\>271, 261-\>260, 265, 275 |
| src/inspect\_sandboxes/daytona/\_daytona.py        |      167 |       15 |       44 |        5 |     90% |75, 85, 108, 181, 206, 219-220, 233-\>252, 246-250, 301-305 |
| src/inspect\_sandboxes/daytona/\_dind\_env.py      |      159 |       21 |       44 |       12 |     82% |92-\>108, 101-104, 115, 134, 154-\>158, 186, 193-194, 244, 277-\>280, 281, 293, 321-326, 361-364, 373 |
| src/inspect\_sandboxes/daytona/\_dind\_project.py  |      224 |       94 |       68 |        5 |     52% |142-\>154, 149-150, 170-186, 204-285, 326-356, 392-\>399, 448, 459, 474-475, 493-\>exit, 495-496, 507-518 |
| src/inspect\_sandboxes/daytona/\_retry.py          |       26 |        4 |        6 |        1 |     78% | 60, 71-74 |
| src/inspect\_sandboxes/daytona/\_sandbox\_utils.py |       50 |        6 |       10 |        1 |     88% |60-61, 81, 89, 107, 112 |
| src/inspect\_sandboxes/daytona/\_single\_env.py    |      119 |       23 |       20 |        1 |     80% |148-\>151, 182-183, 190-196, 203-207, 217-219, 225-229, 235-237 |
| src/inspect\_sandboxes/modal/\_\_init\_\_.py       |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/modal/\_compose.py          |      119 |        5 |       72 |        3 |     96% |212-213, 220-221, 263-\>268, 264-\>263, 269 |
| src/inspect\_sandboxes/modal/\_modal.py            |      271 |       54 |       64 |        6 |     79% |100, 110, 263-268, 330-343, 374, 384-387, 400, 404-405, 427-430, 437-438, 451, 458, 463-466, 492-495, 499-501, 505-520, 524 |
| **TOTAL**                                          | **1310** |  **241** |  **412** |   **43** | **80%** |           |


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