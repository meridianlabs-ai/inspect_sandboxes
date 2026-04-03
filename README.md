# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/meridianlabs-ai/inspect_sandboxes/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/inspect\_sandboxes/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/\_registry.py               |        2 |        2 |        0 |        0 |      0% |       2-3 |
| src/inspect\_sandboxes/\_util/\_\_init\_\_.py      |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/\_util/compose.py           |       30 |        0 |       12 |        1 |     98% |   22-\>21 |
| src/inspect\_sandboxes/\_version.py                |       11 |       11 |        0 |        0 |      0% |      3-24 |
| src/inspect\_sandboxes/daytona/\_\_init\_\_.py     |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/daytona/\_compose.py        |      100 |        6 |       62 |        8 |     91% |109-\>108, 111-112, 219, 224, 228-\>239, 229-\>228, 233, 243 |
| src/inspect\_sandboxes/daytona/\_daytona.py        |      163 |       15 |       44 |        5 |     89% |73, 83, 106, 170, 195, 208-209, 222-\>241, 235-239, 290-294 |
| src/inspect\_sandboxes/daytona/\_dind\_env.py      |      158 |       21 |       44 |       12 |     82% |88-\>104, 97-100, 111, 126, 144-\>148, 176, 183-184, 230, 263-\>266, 267, 279, 307-312, 347-350, 359 |
| src/inspect\_sandboxes/daytona/\_dind\_project.py  |      224 |       95 |       68 |        6 |     52% |142-\>154, 149-150, 170-186, 204-285, 326-356, 385-\>392, 393, 439, 450, 465-466, 484-\>exit, 486-487, 498-509 |
| src/inspect\_sandboxes/daytona/\_retry.py          |       26 |        4 |        6 |        1 |     78% | 60, 71-74 |
| src/inspect\_sandboxes/daytona/\_sandbox\_utils.py |       48 |        5 |        8 |        0 |     91% |60-61, 85, 103, 108 |
| src/inspect\_sandboxes/daytona/\_single\_env.py    |      121 |       23 |       22 |        1 |     80% |150-\>153, 184-185, 192-198, 205-209, 219-221, 227-231, 237-239 |
| src/inspect\_sandboxes/modal/\_\_init\_\_.py       |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/modal/\_compose.py          |      119 |        5 |       72 |        3 |     96% |212-213, 220-221, 263-\>268, 264-\>263, 269 |
| src/inspect\_sandboxes/modal/\_modal.py            |      270 |       54 |       64 |        6 |     79% |98, 108, 260-265, 327-340, 369, 379-382, 395, 399-400, 422-425, 432-433, 446, 453, 458-461, 487-490, 494-496, 500-515, 519 |
| **TOTAL**                                          | **1272** |  **241** |  **402** |   **43** | **79%** |           |


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