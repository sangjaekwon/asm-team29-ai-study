from .schema import Agent3Request, Agent3Response

class Agent3Service:
    def classify(self, request: Agent3Request) -> Agent3Response:
        return Agent3Response(
                recipe_type="Korean"
                )
