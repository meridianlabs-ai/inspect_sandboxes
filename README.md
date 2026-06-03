# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/meridianlabs-ai/inspect_sandboxes/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/inspect\_sandboxes/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/\_registry.py               |        3 |        3 |        0 |        0 |      0% |       2-4 |
| src/inspect\_sandboxes/\_util/\_\_init\_\_.py      |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/\_util/compose.py           |       39 |        2 |       20 |        3 |     92% |22-\>21, 81-\>80, 83-84 |
| src/inspect\_sandboxes/\_util/dind\_compose.py     |       61 |        8 |       28 |        9 |     81% |68, 71, 76-81, 83-\>85, 105, 108, 111, 113-\>103, 116 |
| src/inspect\_sandboxes/\_util/naming.py            |       24 |        0 |        8 |        0 |    100% |           |
| src/inspect\_sandboxes/\_version.py                |       11 |       11 |        0 |        0 |      0% |      3-24 |
| src/inspect\_sandboxes/daytona/\_\_init\_\_.py     |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/daytona/\_compose.py        |       99 |        4 |       56 |        6 |     94% |234, 239, 243-\>254, 244-\>243, 248, 258 |
| src/inspect\_sandboxes/daytona/\_daytona.py        |      167 |       15 |       44 |        5 |     90% |75, 85, 108, 181, 206, 219-220, 233-\>252, 246-250, 301-305 |
| src/inspect\_sandboxes/daytona/\_dind\_env.py      |      160 |       21 |       44 |       12 |     82% |93-\>109, 102-105, 116, 135, 155-\>159, 187, 194-195, 245, 278-\>281, 282, 294, 322-327, 362-365, 374 |
| src/inspect\_sandboxes/daytona/\_dind\_project.py  |      172 |       56 |       40 |        5 |     64% |145-\>157, 152-153, 173-189, 203-226, 247-277, 313-\>320, 369, 382, 397-398, 416-\>exit, 418-419, 430-441 |
| src/inspect\_sandboxes/daytona/\_retry.py          |       26 |        4 |        6 |        1 |     78% | 60, 71-74 |
| src/inspect\_sandboxes/daytona/\_sandbox\_utils.py |       50 |        6 |       10 |        1 |     88% |61-62, 82, 90, 108, 113 |
| src/inspect\_sandboxes/daytona/\_single\_env.py    |      119 |       23 |       20 |        1 |     80% |148-\>151, 182-183, 190-196, 203-207, 217-219, 225-229, 235-237 |
| src/inspect\_sandboxes/e2b/\_\_init\_\_.py         |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/e2b/\_compose.py            |       87 |        1 |       42 |        4 |     96% |125, 178-\>180, 180-\>186, 183-\>186 |
| src/inspect\_sandboxes/e2b/\_dind\_env.py          |      170 |       32 |       46 |       13 |     77% |85-\>101, 94-97, 112-\>115, 141-142, 185, 212-213, 221-\>224, 225, 237, 245-246, 268-273, 287-288, 293-301, 316-321, 340-343, 354, 360, 363, 375 |
| src/inspect\_sandboxes/e2b/\_dind\_project.py      |      158 |       45 |       38 |        4 |     66% |162-\>173, 168-169, 187-212, 224-242, 357, 373, 385-386, 405, 415-425 |
| src/inspect\_sandboxes/e2b/\_e2b.py                |      225 |       24 |       84 |       21 |     85% |62-\>64, 77, 87, 104-\>exit, 106, 117, 122-\>exit, 147, 159, 164-\>177, 184-\>188, 187, 225-\>228, 231, 269, 272, 301-308, 314-315, 316-\>342, 332-334, 337-\>321, 343, 359, 372-\>369, 395-399 |
| src/inspect\_sandboxes/e2b/\_retry.py              |       32 |        2 |        8 |        2 |     90% |    48, 91 |
| src/inspect\_sandboxes/e2b/\_single\_env.py        |      124 |        7 |       18 |        1 |     94% |132-133, 172, 194, 238-239, 253 |
| src/inspect\_sandboxes/e2b/\_template.py           |       37 |        0 |        2 |        0 |    100% |           |
| src/inspect\_sandboxes/modal/\_\_init\_\_.py       |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/modal/\_compose.py          |      119 |        5 |       72 |        3 |     96% |212-213, 220-221, 263-\>268, 264-\>263, 269 |
| src/inspect\_sandboxes/modal/\_modal.py            |      271 |       54 |       64 |        6 |     79% |100, 110, 263-268, 330-343, 374, 384-387, 400, 404-405, 427-430, 437-438, 451, 458, 463-466, 492-495, 499-501, 505-520, 524 |
| **TOTAL**                                          | **2154** |  **323** |  **650** |   **97** | **83%** |           |


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