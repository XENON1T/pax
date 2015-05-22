from celery import Celery
from pax import core
from pax.datastructure import Event

# Specify mongodb host and datababse to connect to
BROKER_URL = 'mongodb://daqeb0:27017/jobs'

mypax = core.Processor(config_names='eventbuilder_worker')

app = Celery('EOD_TASKS',
             broker=BROKER_URL,
             backend=BROKER_URL)


# todo: make class
@app.task
def process(primers):
    mypax = core.Processor(config_names='eventbuilder_worker')

    for primer in primers:
        event = Event(**primer)

        mypax.process_event(event)
