from typing import Literal


class Author:
    username: str
    institution: str

    def __init__(self, username: str):
        raise NotImplementedError()


class GithubRepo:
    url: str
    owner: str
    repository: str

    def __init__(self, owner: str, repository: str):
        raise NotImplementedError()


class ModelBase:
    name: str
    description: str
    author: Author
    respository: GithubRepo
    disease: Literal["dengue", "zika", "chikungunya"]
    type: Literal["time", "spatial"]
    time_resolution: Literal["day", "week", "month", "year"]
    ADM_level: Literal[0, 1, 2, 3]

    def __init__(self, id):
        print(id)
