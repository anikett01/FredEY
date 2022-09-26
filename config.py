
import os

class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "1abc96f5-6c19-4cc7-9cd7-de1d4408fa30") #Enter your Microsoft App Id here (Azure Bot App ID)
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "sba8Q~h83G2~M.jQ1SVmVNu3eDqck6Hz1BkBrcvV") #Enter your Microsoft App Password here
    #EXPIRE_AFTER_SECONDS = os.environ.get("ExpireAfterSeconds", 60)
