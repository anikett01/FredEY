
import os

class DefaultConfig:
    """ Bot Configuration """

    PORT = 8000
    APP_ID = os.environ.get("MicrosoftAppId", "89ce46e0-434c-44f8-8505-805387539074") #Enter your Microsoft App Id here (Azure Bot App ID)
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "V-X8Q~zqfXgeO_61VxPMy6eXDkSWSGr2suGuPaFY") #Enter your Microsoft App Password here
    #EXPIRE_AFTER_SECONDS = os.environ.get("ExpireAfterSeconds", 60)
