"""Write to Amazon dynamoDB
"""
import boto3

from pax import plugin


class WriteDynamoDB(plugin.OutputPlugin):
    do_input_check = False
    do_output_check = False

    def startup(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table('processed')

    def write_event(self, event):
        doc = event.to_dict(convert_numpy_arrays_to='list',
                            nan_to_none=True,
                            fields_to_ignore=self.config['fields_to_ignore'],
                            use_decimal=True)

        doc['peaks'] = [peak for peak in doc['peaks'] if peak['area'] > 100]

        self.table.put_item(Item=doc)
