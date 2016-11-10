"""Output to Apache's messagepack format"""

import msgpack
from pax import datastructure
from pax.FolderIO import WriteZippedEncoder, ReadZippedDecoder


class EncodeMessagePack(WriteZippedEncoder):

    def encode_event(self, event):
        event_dict = event.to_dict(fields_to_ignore=self.config['fields_to_ignore'],
                                   convert_numpy_arrays_to='bytes')
        return msgpack.packb(event_dict, use_single_float=self.config.get('use_single_float', False))


class DecodeMessagePack(ReadZippedDecoder):

    def decode_event(self, event_msgpack):
        event_dict = msgpack.unpackb(event_msgpack)
        # MessagePack returns byte keys, which we can't pass as keyword arguments
        # Unfortunately we have to duplicate this code in data_model too
        event_dict = {k.decode('ascii'): v for k, v in event_dict.items()}
        return datastructure.Event(**event_dict)
