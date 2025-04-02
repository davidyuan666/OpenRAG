# -*- coding = utf-8 -*-
# @time:2024/7/29 10:15
# Author:david yuan
# @File:base.py
# @Software:VeSync


from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from vagents.vagentic.document import Document


class Distance(str, Enum):
    cosine = "cosine"
    l2 = "l2"
    max_inner_product = "max_inner_product"


class VectorDb(ABC):
    """Base class for managing Vector Databases"""

    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def doc_exists(self, document: Document) -> bool:
        raise NotImplementedError

    @abstractmethod
    def name_exists(self, name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def insert(self, documents: List[Document]) -> None:
        raise NotImplementedError

    def upsert_available(self) -> bool:
        return False

    @abstractmethod
    def upsert(self, documents: List[Document]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Document]:
        raise NotImplementedError

    @abstractmethod
    def delete(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def optimize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> bool:
        raise NotImplementedError
