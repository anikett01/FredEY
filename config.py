
import os

class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "27683de7-f005-409e-85ca-8aeea2db0b98") #Enter your Microsoft App Id here (Azure Bot App ID)
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "ywH8Q~OIvL_61ZdZy1RZ5nkapbNiNGaDA9WxwaY4") #Enter your Microsoft App Password here
    #EXPIRE_AFTER_SECONDS = os.environ.get("ExpireAfterSeconds", 60)

    
