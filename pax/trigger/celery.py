from celery import Celery
BROKER_URL = 'mongodb://127.0.0.1:27017/jobs'

app = Celery('trigger',
             broker=BROKER_URL,
             backend=BROKER_URL,
             include=['pax.trigger.tasks'])

# Optional configuration, see the application user guide.
app.conf.update(
    CELERY_TASK_RESULT_EXPIRES=3600,
)

if __name__ == '__main__':
    app.start()
