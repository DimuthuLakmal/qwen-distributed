from fastapi import Request

from src.inference.router.service import RouterService


def get_router_service(request: Request) -> RouterService:
    """
    Return the RouterService created once during router startup (see
    `src/inference/router/app.py`'s lifespan).
    """

    return request.app.state.router_service
