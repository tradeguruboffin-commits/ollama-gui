import datetime
def run(text):
    if "time" in text:
        return str(datetime.datetime.now())
