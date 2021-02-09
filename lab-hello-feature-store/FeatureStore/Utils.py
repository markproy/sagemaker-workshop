import sagemaker
from sagemaker.session import Session
from sagemaker import get_execution_role
from sagemaker.feature_store.feature_group import FeatureGroup

import boto3
import json
from time import sleep
import pandas as pd
import os
import time

#print(sagemaker.__version__)

role = get_execution_role()
boto_session = boto3.Session()
region = boto_session.region_name
account_id = boto3.client("sts").get_caller_identity()["Account"]

s3_client = boto3.client('s3', region_name=region)

sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)
featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region)

feature_store_session = Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client,
    sagemaker_featurestore_runtime_client=featurestore_runtime
)

sm_sess = sagemaker.Session()
default_bucket = sm_sess.default_bucket() 

def _wait_for_feature_group_deletion_complete(feature_group_name):
    region = boto3.Session().region_name
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)

    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    
    while True:
        try:
            status = feature_group.describe().get("FeatureGroupStatus")
            print("Waiting for Feature Group Deletion")
            time.sleep(5)
        except:
            break
    return

def _wait_for_feature_group_creation_complete(feature_group_name):
    region = boto3.Session().region_name
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)

    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)
    
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for Feature Group Creation")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully created.")
    
def _df_to_feature_defs(df):
    feature_definitions = []

    for col, col_type in zip(df.columns, df.dtypes):
        feature = {'FeatureName': col}
        
        if col_type == 'float64':
            feature['FeatureType'] = 'Fractional'
        elif col_type == 'int64':
            feature['FeatureType'] = 'Integral'
        else:
            feature['FeatureType'] = 'String'
        
        feature_definitions.append(feature)    
    return feature_definitions

def _escape_tag_chars(in_str):
    escaped_str = in_str.replace('$', '_D_')
    escaped_str = escaped_str.replace('?', '_Q_')
    return escaped_str

def _unescape_tag_chars(in_str):
    unescaped_str = in_str.replace('_D_', '$')
    unescaped_str = unescaped_str.replace('_Q_', '?')
    return unescaped_str

def create_fg_from_df(fg_name, df, id_name='Id', event_time_name='UpdateTime', tags=None,
                      online=True, s3_uri=None):
    ## TODO: add support for passing in Description and Tags
    
    if not id_name in df.columns:
        print(f'invalid id column name: {id_name}')
        return
    if not event_time_name in df.columns:
        print(f'invalid event time column name: {event_time_name}')
        return
    
    if s3_uri is None:
        s3_uri = f's3://{default_bucket}/offline-store'
    
    other_args = {}
    if s3_uri is not None:
        other_args['OfflineStoreConfig'] = {'S3StorageConfig': {'S3Uri': s3_uri}}
        
    if tags is not None:
        tags_as_kv_array = []
        for k, v in tags.items():
            curr_kv = {'Key': k, 'Value': _escape_tag_chars(v)}
            tags_as_kv_array.append(curr_kv)
        other_args['Tags'] = tags_as_kv_array

    resp = sagemaker_client.create_feature_group(
            FeatureGroupName = fg_name,
            RecordIdentifierFeatureName = id_name,
            EventTimeFeatureName = event_time_name,
            FeatureDefinitions = _df_to_feature_defs(df),
            OnlineStoreConfig = {'EnableOnlineStore': online},
            RoleArn = role,
            **other_args)
    
    _wait_for_feature_group_creation_complete(fg_name)
    return 

def get_tags(fg_name):
    fg_arn = describe_feature_group(fg_name)['FeatureGroupArn']
    resp = sagemaker_client.list_tags(ResourceArn=fg_arn)
    tags_kv_array = resp['Tags']
    tags = {}
    for kv in tags_kv_array:
        k = kv['Key']
        v = kv['Value']
        tags[k] = _unescape_tag_chars(v)
        
    return tags

def list_feature_groups(name_contains=None):
    if name_contains is None:
        resp = sagemaker_client.list_feature_groups()
    else:
        resp = sagemaker_client.list_feature_groups(NameContains=name_contains)
    return resp['FeatureGroupSummaries']

def describe_feature_group(fg_name):
    return sagemaker_client.describe_feature_group(FeatureGroupName=fg_name)

def get_offline_store_url(fg_name):
    fg_s3_uri = ''
    has_offline_store = True
    offline_store_config = {}
    try:
        offline_store_config = describe_feature_group(fg_name)['OfflineStoreConfig']
    except:
        has_offline_store = False
        return fg_s3_uri

    base_s3_uri = offline_store_config['S3StorageConfig']['S3Uri']
    base_offline_prefix = '/'.join(base_s3_uri.split('/')[3:])
    offline_prefix = f'{base_offline_prefix}/{account_id}/sagemaker/{region}/offline-store/{fg_name}'
    s3_bucket_name = base_s3_uri.split('/')[2]
    
    return f'https://s3.console.aws.amazon.com/s3/buckets/{s3_bucket_name}?region={region}&prefix={offline_prefix}/data/'

def get_glue_table_url(fg_name):
    _data_catalog_config = describe_feature_group(fg_name)['OfflineStoreConfig']['DataCatalogConfig']
    _table = _data_catalog_config['TableName']
    _database = _data_catalog_config['Database']

    return f'https://console.aws.amazon.com/glue/home?region={region}#table:catalog={account_id};name={_table};namespace={_database}'

def download_sample_offline_file(fg_name):
    fg_s3_uri = ''
    has_offline_store = True
    offline_store_config = {}
    try:
        offline_store_config = describe_feature_group(fg_name)['OfflineStoreConfig']
    except:
        has_offline_store = False
        return fg_s3_uri

    base_s3_uri = offline_store_config['S3StorageConfig']['S3Uri']
    bucket = base_s3_uri.split('s3://')[1].split('/')[0]
    prefix = base_s3_uri.replace(f's3://{bucket}/', '')

    s3_client = boto3.client('s3')
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    key_to_download = ''
    for obj_key in resp['Contents']:
        curr_key = obj_key['Key']
        if '.parquet' in curr_key:
            key_to_download = curr_key
            break
            
    filename = key_to_download.split('/')[-1]
    s3_client.download_file(default_bucket, key_to_download, filename)
    return filename

def delete_feature_group(fg_name, delete_s3=True):
    ## TODO: wait until it is fully deleted b4 returning
    ## TODO: properly handle situation when fg is already gone
    ## TODO: Delete Glue table if one was created automatically

    has_offline_store = True
    try:
        describe_feature_group(fg_name)['OfflineStoreConfig']
    except:
        has_offline_store = False
        pass

    if has_offline_store:
        offline_store_config = describe_feature_group(fg_name)['OfflineStoreConfig']
        if not offline_store_config['DisableGlueTableCreation']:
            table_name = offline_store_config['DataCatalogConfig']['TableName']
            catalog_id = offline_store_config['DataCatalogConfig']['Catalog']
            database_name = offline_store_config['DataCatalogConfig']['Database']
        
#         glue_client = boto3.client('glue')

#         try:
#             glue_client.delete_table(
#                 CatalogId=catalog_id,
#                 DatabaseName=database_name,
#                 Name=table_name
#             )    
#         except:
#             # An error occurred (AccessDeniedException) when calling the DeleteTable operation: 
#             # Cross account access is not supported for account that hasn't imported Athena catalog to Glue.
#             print('Failed to delete Glue table.')
#             print('See https://docs.aws.amazon.com/athena/latest/ug/glue-upgrade.html')
            
        
    # Delete s3 objects from offline store for this FG
    if delete_s3 and has_offline_store:
        s3_uri = describe_feature_group(fg_name)['OfflineStoreConfig']['S3StorageConfig']['S3Uri']
        base_offline_prefix = '/'.join(s3_uri.split('/')[3:])
        offline_prefix = f'{base_offline_prefix}/{account_id}/sagemaker/{region}/offline-store/{fg_name}'
        s3_bucket_name = s3_uri.split('/')[2]
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(s3_bucket_name)
        coll = bucket.objects.filter(Prefix=offline_prefix)
        print(f'Deleting all s3 objects in prefix: {offline_prefix} in bucket {s3_bucket_name}')
        resp = coll.delete()
    
    resp = None
    try:
        resp = sagemaker_client.delete_feature_group(FeatureGroupName=fg_name)
    except:
        pass
    
    _wait_for_feature_group_deletion_complete(fg_name)
    return 

def ingest_from_df(fg_name, df):
    fg = FeatureGroup(name=fg_name, sagemaker_session=feature_store_session)
    fg.ingest(data_frame=df, max_workers=2, wait=True)
    
def _record_to_dict(rec, feature_types):
    tmp_dict = {}
    for f in rec:
        feature_name = f['FeatureName']
        string_feature_val = f['ValueAsString']
        feature_type = feature_types[feature_name]
        
        if feature_type == 'Integral':
            tmp_dict[f['FeatureName']] = int(string_feature_val)
        elif feature_type == 'Fractional':
            tmp_dict[f['FeatureName']] = float(string_feature_val)
        else:
            tmp_dict[f['FeatureName']] = string_feature_val

    return tmp_dict

def delete_record(fg_name, record_id, event_time):
    results = []
    
    resp = featurestore_runtime.delete_record(FeatureGroupName=fg_name, 
                                              RecordIdentifierValueAsString=str(record_id),
                                              EventTime=event_time)
    return

def get_latest_feature_values(fg_name, id_value_list):
    feature_defs = describe_feature_group(fg_name)['FeatureDefinitions']
    feature_types = {}
    for fd in feature_defs:
        feature_types[fd['FeatureName']] = fd['FeatureType']
        
    results = []
    
    for curr_id in id_value_list:
        record_identifier_value = str(curr_id)
        resp = featurestore_runtime.get_record(FeatureGroupName=fg_name, 
                                               RecordIdentifierValueAsString=record_identifier_value)
        try:
            curr_record = _record_to_dict(resp['Record'], feature_types)
            results.append(curr_record)
        except:
            pass
    return results

def _run_query(query_string, tmp_uri, database, verbose=True):
    athena = boto3.client('athena')

    # submit the Athena query
    if verbose:
        print('Running query:\n ' + query_string)
    query_execution = athena.start_query_execution(
        QueryString=query_string,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': tmp_uri}
    )

    # wait for the Athena query to complete
    query_execution_id = query_execution['QueryExecutionId']
    query_state = athena.get_query_execution(QueryExecutionId=query_execution_id)['QueryExecution']['Status']['State']
    while (query_state != 'SUCCEEDED' and query_state != 'FAILED'):
        sleep(2)
        query_state = athena.get_query_execution(QueryExecutionId=query_execution_id)['QueryExecution']['Status']['State']
    
    if query_state == 'FAILED':
        print(athena.get_query_execution(QueryExecutionId=query_execution_id))
        failure_reason = athena.get_query_execution(QueryExecutionId=query_execution_id)['QueryExecution']['Status']['StateChangeReason']
        print(failure_reason)
        df = pd.DataFrame()
        return df
    else:
        ## TODO: fix this to avoid hardcoding prefix
        results_file_prefix = f'offline-store/query_results/{query_execution_id}.csv'

        # Prepare query results for training.
        filename = 'query_results.csv'
        results_bucket = (tmp_uri.split('//')[1]).split('/')[0]
        s3_client.download_file(results_bucket, results_file_prefix, filename)
        df = pd.read_csv('query_results.csv')
        os.remove('query_results.csv')
        
        s3_client.delete_object(Bucket=results_bucket, Key=results_file_prefix)
        s3_client.delete_object(Bucket=results_bucket, Key=results_file_prefix + '.metadata')
        return df

def _get_offline_details(fg_name, s3_uri=None):
    _data_catalog_config = describe_feature_group(fg_name)['OfflineStoreConfig']['DataCatalogConfig']
    _table = _data_catalog_config['TableName']
    _database = _data_catalog_config['Database']

    if s3_uri is None:
        s3_uri = f's3://{default_bucket}/offline-store'
    _tmp_uri = f'{s3_uri}/query_results/'
    return _table, _database, _tmp_uri

def get_historical_record_count(fg_name, s3_uri=None):
    _table, _database, _tmp_uri = _get_offline_details(fg_name, s3_uri)
    _query_string = f'SELECT COUNT(*) FROM "' +_table+ f'"'
    _tmp_df = _run_query(_query_string, _tmp_uri, _database, verbose=False)
    return _tmp_df.iat[0, 0]
    
def get_historical_offline_feature_values(fg_name, record_ids=None, feature_names=None, s3_uri=None):
    _table, _database, _tmp_uri = _get_offline_details(fg_name, s3_uri)
    
    # construct an Athena query
    
    id_feature_name = describe_feature_group(fg_name)['RecordIdentifierFeatureName']
    time_feature_name = describe_feature_group(fg_name)['EventTimeFeatureName']

    if feature_names is None:
        feature_name_string = '*'
    else:
        feature_name_string = ','.join(feature_names)
    
    if record_ids is None:
        where_clause = ''
    else:
        id_list_string = ','.join(str(x) for x in record_ids)
        where_clause = f' WHERE {id_feature_name} IN ({id_list_string})'
    
    _query_string = f'SELECT {feature_name_string} FROM "' +_table+ f'" {where_clause}'
    
    return _run_query(_query_string, _tmp_uri, _database)
    
def get_latest_offline_feature_values(fg_name, record_ids=None, feature_names=None, s3_uri=None):
    _table, _database, _tmp_uri = _get_offline_details(fg_name, s3_uri)

    # construct an Athena query
    
    id_feature_name = describe_feature_group(fg_name)['RecordIdentifierFeatureName']
    time_feature_name = describe_feature_group(fg_name)['EventTimeFeatureName']

    if feature_names is None:
        feature_name_string = '*'
    else:
        feature_name_string = ','.join(feature_names)
    
    if record_ids is None:
        where_clause = ''
    else:
        id_list_string = ','.join(str(x) for x in record_ids)
        where_clause = f' WHERE {id_feature_name} IN ({id_list_string})'
    
    _subquery = f'SELECT *, dense_rank() OVER (PARTITION BY {id_feature_name} ' + \
                f'ORDER BY {time_feature_name} DESC, Api_Invocation_Time DESC, write_time DESC) AS rank ' + \
                f'FROM "' +_table+ f'" {where_clause}'
    _query_string = f'SELECT {feature_name_string} FROM ({_subquery}) WHERE rank = 1 AND NOT is_deleted'

    return _run_query(_query_string, _tmp_uri, _database)

def get_offline_feature_values_as_of(fg_name, as_of, record_ids=None, feature_names=None, s3_uri=None):
    _table = describe_feature_group(fg_name)['OfflineStoreConfig']['DataCatalogConfig']['TableName']
    _database = describe_feature_group(fg_name)['OfflineStoreConfig']['DataCatalogConfig']['Database']
    if s3_uri is None:
        s3_uri = f's3://{default_bucket}/offline-store'

    _tmp_uri = f'{s3_uri}/query_results/'

    # construct an Athena query
    
    id_feature_name = describe_feature_group(fg_name)['RecordIdentifierFeatureName']
    time_feature_name = describe_feature_group(fg_name)['EventTimeFeatureName']

    if feature_names is None:
        feature_name_string = '*'
    else:
        feature_name_string = ','.join(feature_names)
    
    if record_ids is None:
        where_clause = ''
    else:
        id_list_string = ','.join(str(x) for x in record_ids)
        where_clause = f' WHERE {id_feature_name} IN ({id_list_string})'
    
    ## TODO: resolve issue with Presto and iso 8601 timestamps. partial solution provided by from_iso8601_timestamp
    ##  https://aws.amazon.com/premiumsupport/knowledge-center/query-table-athena-timestamp-empty/
    _subquery = f'SELECT *, dense_rank() OVER (PARTITION BY {id_feature_name} ' + \
                f'ORDER BY {time_feature_name} DESC, Api_Invocation_Time DESC, write_time DESC) AS rank ' + \
                f'FROM "' +_table+ f'" {where_clause}' + \
                f"WHERE {time_feature_name} <= '{as_of.upper()}'"
##                f"WHERE {time_feature_name} <= TIMESTAMP '{as_of.upper()}'"
    _query_string = f'SELECT {feature_name_string} FROM ({_subquery}) WHERE rank = 1 AND NOT is_deleted'

    return _run_query(_query_string, _tmp_uri, _database)