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
| src/inspect\_sandboxes/daytona/\_daytona.py    |      272 |       32 |       66 |        9 |     87% |87, 97, 118, 169, 176-178, 195, 208-209, 222->241, 235-237, 291-295, 336, 375, 381-385, 397->400, 429-430, 454, 494-495, 504-505, 520-523 |
| src/inspect\_sandboxes/modal/\_\_init\_\_.py   |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/modal/\_compose.py      |      114 |        5 |       68 |        3 |     96% |205-206, 213-214, 256->261, 257->256, 262 |
| src/inspect\_sandboxes/modal/\_modal.py        |      263 |       57 |       64 |        6 |     78% |75, 85, 237-242, 303-316, 321-322, 345, 354-358, 371, 375-376, 399-402, 409-410, 423, 430, 435-438, 459-462, 465-467, 470-485, 489 |
| **TOTAL**                                      |  **763** |  **113** |  **256** |   **25** | **85%** |           |


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