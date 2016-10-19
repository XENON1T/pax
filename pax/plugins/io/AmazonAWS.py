"""Write to Amazon dynamoDB
"""
import boto3

from pax import plugin


class WriteDynamoDB(plugin.OutputPlugin):
    do_input_check = False
    do_output_check = False

    def startup(self):
        self.dynamodb = boto3.resource('dynamodb')
        
    def write_event(self, event):
        self.table.put_item(Item=event.to_dict(
            fields_to_ignore=self.config['fields_to_ignore']))
