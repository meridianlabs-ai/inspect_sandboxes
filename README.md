# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/meridianlabs-ai/inspect_sandboxes/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                               |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/inspect\_sandboxes/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/\_registry.py               |        3 |        3 |        0 |        0 |      0% |       2-4 |
| src/inspect\_sandboxes/\_util/\_\_init\_\_.py      |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/\_util/compose.py           |       72 |        2 |       30 |        3 |     95% |23-\>22, 82-\>81, 84-85 |
| src/inspect\_sandboxes/\_util/dind\_compose.py     |       61 |        8 |       28 |        9 |     81% |68, 71, 76-81, 83-\>85, 105, 108, 111, 113-\>103, 116 |
| src/inspect\_sandboxes/\_util/naming.py            |       24 |        0 |        8 |        0 |    100% |           |
| src/inspect\_sandboxes/\_version.py                |       11 |       11 |        0 |        0 |      0% |      3-24 |
| src/inspect\_sandboxes/daytona/\_\_init\_\_.py     |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/daytona/\_compose.py        |      117 |        4 |       68 |        7 |     94% |146-\>138, 283, 288, 292-\>303, 293-\>292, 297, 307 |
| src/inspect\_sandboxes/daytona/\_daytona.py        |      183 |       21 |       48 |        7 |     87% |84, 94, 117, 197, 222, 235-236, 249-\>270, 256, 264-268, 282-287, 332-336 |
| src/inspect\_sandboxes/daytona/\_dind\_env.py      |      160 |       21 |       44 |       12 |     82% |93-\>109, 102-105, 116, 135, 155-\>159, 187, 194-195, 245, 278-\>281, 282, 294, 322-327, 362-365, 374 |
| src/inspect\_sandboxes/daytona/\_dind\_project.py  |      174 |       56 |       40 |        5 |     64% |146-\>158, 153-154, 174-190, 204-227, 249-279, 315-\>322, 371, 384, 399-400, 418-\>exit, 420-421, 432-443 |
| src/inspect\_sandboxes/daytona/\_retry.py          |       26 |        4 |        6 |        1 |     78% | 64, 75-78 |
| src/inspect\_sandboxes/daytona/\_sandbox\_utils.py |      127 |       15 |       32 |        4 |     88% |77-78, 110-111, 147-150, 204, 207-\>209, 232, 237-238, 267, 285, 290 |
| src/inspect\_sandboxes/daytona/\_single\_env.py    |      142 |       28 |       26 |        2 |     80% |159-\>162, 208-214, 218-224, 245-246, 253-259, 266-270, 280-282, 288-292, 298-300 |
| src/inspect\_sandboxes/e2b/\_\_init\_\_.py         |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/e2b/\_compose.py            |      105 |        1 |       54 |        5 |     96% |148-\>140, 173, 226-\>228, 228-\>234, 231-\>234 |
| src/inspect\_sandboxes/e2b/\_dind\_env.py          |      171 |       32 |       46 |       13 |     77% |86-\>102, 95-98, 113-\>116, 142-143, 188, 215-216, 224-\>227, 228, 242, 250-251, 273-278, 294-295, 300-308, 323-328, 347-350, 361, 367, 370, 382 |
| src/inspect\_sandboxes/e2b/\_dind\_project.py      |      159 |       45 |       38 |        4 |     66% |163-\>174, 169-170, 188-213, 225-247, 362, 378, 390-391, 410, 420-430 |
| src/inspect\_sandboxes/e2b/\_e2b.py                |      240 |       28 |       90 |       22 |     84% |64-\>66, 79, 89, 106-\>exit, 108, 119, 124-\>exit, 149, 161, 166-\>179, 186-\>190, 189, 232-\>235, 238, 276, 279-282, 298, 303-304, 327-334, 340-341, 342-\>368, 358-360, 363-\>347, 369, 385, 398-\>395, 421-425 |
| src/inspect\_sandboxes/e2b/\_retry.py              |       32 |        2 |        8 |        2 |     90% |    48, 91 |
| src/inspect\_sandboxes/e2b/\_single\_env.py        |      137 |        7 |       22 |        1 |     95% |144-145, 184, 237, 288-289, 305 |
| src/inspect\_sandboxes/e2b/\_template.py           |       37 |        0 |        2 |        0 |    100% |           |
| src/inspect\_sandboxes/modal/\_\_init\_\_.py       |        0 |        0 |        0 |        0 |    100% |           |
| src/inspect\_sandboxes/modal/\_compose.py          |      144 |        5 |       90 |        4 |     96% |149-\>134, 287-288, 295-296, 338-\>343, 339-\>338, 344 |
| src/inspect\_sandboxes/modal/\_modal.py            |      292 |       57 |       72 |        6 |     80% |110, 120, 274-279, 341-354, 385, 395-398, 411, 415-416, 438-441, 448-449, 478-482, 512, 519, 524-527, 553-556, 560-562, 566-581, 585 |
| **TOTAL**                                          | **2417** |  **350** |  **752** |  **107** | **84%** |           |


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