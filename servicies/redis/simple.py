import redis
import json
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid


### Main redis config
class RedisConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    ttl: int = 300


# Base class for components . default factory to give a deafult value
class BaseComponent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class AudioComponent(BaseComponent):
    wave_domain: float
    bit_rate: float


class VideoComponent(BaseComponent):
    frames: int


class ImageComponent(BaseComponent):
    width: float
    height: float


class FullResponse(BaseModel):
    audio: AudioComponent
    video: VideoComponent
    image: ImageComponent

    @classmethod
    def get_component_names(cls) -> List[str]:
        return ["audio", "video", "image"]

    @classmethod
    def get_component_class(cls, name: str):
        mapping = {
            "audio": AudioComponent,
            "video": VideoComponent,
            "image": ImageComponent,
        }
        return mapping.get(name)


class RedisWrapper:
    def __init__(self, redis_config: RedisConfig):
        self.redis_config = redis_config
        self.r = redis.Redis(
            host=redis_config.host, port=redis_config.port, db=redis_config.db
        )
        self.components = FullResponse.get_component_names()

    def add_component(self, id: str, component_type: str, data: BaseComponent):
        if component_type not in self.components:
            raise ValueError(f"Wrong component: {component_type}")

        if self.r.hexists(id, component_type):
            raise ValueError(f"Component {component_type} already exists for {id}")

        self.r.hset(id, component_type, data.model_dump_json())
        self.r.expire(id, self.redis_config.ttl)

    def is_complete(self, id: str) -> bool:
        return self.r.hlen(id) == len(self.components)

    def get_all(self, id: str) -> Optional[FullResponse]:
        if not self.is_complete(id):
            print("Not all components added")
            return None

        data = self.r.hgetall(id)

        parsed = {}
        for name in self.components:
            raw = data.get(name.encode())
            if raw:
                component_class = FullResponse.get_component_class(name)
                parsed[name] = component_class.model_validate_json(raw)

        return FullResponse(**parsed)


if __name__ == "__main__":
    config = RedisConfig()
    wrapper = RedisWrapper(config)

    request_id = str(uuid.uuid4())

    wrapper.add_component(
        request_id, "audio", AudioComponent(wave_domain=44.1, bit_rate=320.0)
    )
    wrapper.add_component(request_id, "video", VideoComponent(frames=30))
    wrapper.add_component(
        request_id, "image", ImageComponent(width=1920.0, height=1080.0)
    )

    response = wrapper.get_all(request_id)
    if response:
        print(response.model_dump_json(indent=2))
