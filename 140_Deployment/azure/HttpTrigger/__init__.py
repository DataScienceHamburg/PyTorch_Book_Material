import azure.functions as func
from azure.functions import AsgiMiddleware
from api_app import asgi_app


async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await AsgiMiddleware(asgi_app).handle_async(req, context)


