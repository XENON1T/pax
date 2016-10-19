"""Write to Amazon dynamoDB
"""
import boto3

from pax import plugin


class WriteDynamoDB(plugin.OutputPlugin):
    do_input_check = False
    do_output_check = False

    def startup(self):
        self.dynamodb = boto3.resource('dynamodb')
        # Create the DynamoDB table.
        self.table = self.dynamodb.create_table(TableName='processed',
                                                KeySchema=[
                                                    {
                                                        'AttributeName': 'event_number',
                                                        'KeyType': 'HASH'
                                                    },
                                                    {
                                                        'AttributeName': 'dataset_name',
                                                        'KeyType': 'RANGE'
                                                    }
                                                ],
                                                AttributeDefinitions=[
                                                    {
                                                        'AttributeName': 'event_number',
                                                        'AttributeType': 'I'
                                                    },
                                                    {
                                                        'AttributeName': 'dataset_name',
                                                        'AttributeType': 'S'
                                                    },

                                                ],
                                                ProvisionedThroughput={
                                                    'ReadCapacityUnits': 5,
                                                    'WriteCapacityUnits': 5
                                                }
                                                )

    def write_event(self, event):
        self.table.put_item(Item=event.to_dict(
            fields_to_ignore=self.config['fields_to_ignore']))
