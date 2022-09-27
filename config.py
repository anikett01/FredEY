
import os

class DefaultConfig:
    """ Bot Configuration """

    PORT = 8080
    APP_ID = os.environ.get("MicrosoftAppId", "af261cb0-ae81-49f5-87f0-9ea2cb4d72c5") #Enter your Microsoft App Id here (Azure Bot App ID)
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "YG88Q~ZAj~XX0mQwj8Sgp0AFWLHGem191dmAjdBQ") #Enter your Microsoft App Password here
    #EXPIRE_AFTER_SECONDS = os.environ.get("ExpireAfterSeconds", 60)
