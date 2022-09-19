
import os

class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "") #Enter your Microsoft App Id here (Azure Bot App ID)
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "") #Enter your Microsoft App Password here
    #EXPIRE_AFTER_SECONDS = os.environ.get("ExpireAfterSeconds", 60)
