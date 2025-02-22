import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any


class IStorage(ABC):
    @abstractmethod
    def save_context(self, user_id, context, ttl) -> None: ...

    @abstractmethod
    def load_context(self, user_id) -> Any: ...


class InMemoryStorage(IStorage):
    def __init__(self):
        self.memory = defaultdict(dict)

    def save_context(self, user_id, context, ttl=1800) -> None:
        self.memory[user_id] = context

    def load_context(self, user_id) -> dict:
        return self.memory.get(user_id, {})


class RedisMemoryStorage(IStorage):
    def __init__(self):
        import redis

        self.memory = redis.Redis(host="localhost", port=6379, decode_responses=True)

    def save_context(self, user_id, context, ttl=1800) -> None:
        self.memory.setex(f"user:{user_id}:context", ttl, json.dumps(context))

    def load_context(self, user_id) -> Any:
        data = self.memory.get(f"user:{user_id}:context")
        return json.loads(data) if data else {}
