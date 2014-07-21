__author__ = 'tunnell'


def sizeof_fmt(num):
    """input is bytes"""
    for x in ['B', 'KB', 'MB', 'GB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0
    return "%3.1f %s" % (num, 'TB')


def sampletime_fmt(num):
    """num is in 10s of ns"""
    num *= 10
    for x in ['ns', 'us', 'ms']:
        if num < 1000.0:
            return "%3.1f %s" % (num, x)
        num /= 1000.0
    return "%3.1f %s" % (num, 's')
