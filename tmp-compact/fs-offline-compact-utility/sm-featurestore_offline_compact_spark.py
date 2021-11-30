import pyspark
import sagemaker_pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from datetime import datetime, timedelta
import argparse
import logging
import boto3
import botocore
import time
import os
from typing import Tuple


logger = logging.getLogger('__name__')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

mode_incremental = 'incremental'
mode_full = 'full'
mode_day = 'day'
partition_hour = 'hour'
partition_day = 'day'
client = boto3.client('s3')


def count_objects_at_s3_uri(s3_uri: str) -> Tuple[int, int]:
    s3 = boto3.resource('s3')

    bucket_and_prefix = s3_uri.split('//')[1]
    bucket = bucket_and_prefix.split('/')[0]
    prefix = '/'.join(bucket_and_prefix.split('/')[1:])

    bucket = s3.Bucket(bucket)

    num_obj = 0
    total_size = 0
    
    for object_summary in bucket.objects.filter(Prefix=prefix):
        num_obj += 1
        total_size += object_summary.size

    return num_obj, total_size

def full_traverse(spark_session: SparkSession, base_uri: str, compact_base_uri: str, bucket: str, prefix :str, partition_mode: str):
    """
    When the mode is full, traverse the folders recursively for all years and months and days. Once the day prefix is hit, 
    compact_day method is invoked to process the files within it. 
    """ 
    year = month = day = 0
    src_uri = base_uri
    dst_uri = compact_base_uri
    try:
        folders = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
        if folders and folders.get('CommonPrefixes'):
            for obj in folders.get('CommonPrefixes'):
                prefix = obj.get('Prefix')            
                folder = prefix.split('=')
                if 'day' in folder[-2]:
                    day = int(folder[-1].split('/')[0])
                    src_uri = f'{base_uri}day={day:02d}/'
                    dst_uri = f'{compact_base_uri}day={day:02d}/' 
                    compact_day(spark_session, src_uri, dst_uri, partition_mode)
                elif 'month' in folder[-2]:
                    month = int(folder[-1].split('/')[0])
                    src_uri = f'{base_uri}month={month:02d}/'
                    dst_uri = f'{compact_base_uri}month={month:02d}/'
                elif 'year' in folder[-2]:
                    year = int(folder[-1].split('/')[0])
                    src_uri = f'{base_uri}year={year}/'
                    dst_uri = f'{compact_base_uri}year={year}/'
                full_traverse(spark_session, src_uri, dst_uri, bucket, prefix, partition_mode)
    except botocore.client.ClientError as error:
            logger.info(f'\n No bucket Exception caught: {error.response} for bucket {bucket}')
            
def compact_files(spark_session: SparkSession, src_uri: str, dst_uri: str) -> None:
    """
    This function loads the parquet files into Spark dataframe and repartitions to a single partition and writes back to S3
    """
    num_obj, total_size = count_objects_at_s3_uri(src_uri)
    logger.info(f'\n****  Starting compacting for  ={src_uri} *****')
    logger.info(f'\n****  Num of objects ={num_obj} *****')
    
    if num_obj != 0:
        logger.info(f'\n  Compacting files for ={src_uri}, started with {num_obj:,d} files, {total_size//(1000*1000):,d} MB.')
        try:
            start_time = time.time()
            df = spark_session.read.parquet(src_uri)
            df.rdd.getNumPartitions()
            logger.info(f'took {int(time.time() - start_time):,d} seconds to read')
            df = df.repartition(1)
            start_time = time.time()
            df.write.mode('overwrite').parquet(dst_uri)
            logger.info(f'       took {int(time.time() - start_time):,d} seconds to write')
            compact_num_obj, compact_total_size = count_objects_at_s3_uri(dst_uri)
            logger.info(f'\n  Result: {compact_num_obj:,d} files, {compact_total_size//(1000*1000):,d} MB.')
        except pyspark.sql.utils.AnalysisException as inst:
            error_text = str(inst) 
            if 'Path does not exist' in error_text:
                logger.info(f'    ** input path ={src_uri} does not exist')
            elif 'already exists' in error_text:
                logger.info(f'    *** target path already exists: {dst_uri}')
        except Exception as inst:
            logger.info(type(inst))
            logger.info(inst.args)

    logger.info(f'\n  FINISHED day {src_uri}\n')
    return

def compact_day(spark_session: SparkSession, base_uri: str, compact_base_uri: str, partition_mode: str):   
    """
    This function compacts an entire day worth of files.
    If partition mode == 'day', it compacts all hourly files into a single parquet file for the day.
    If partition mode == 'hour', it compacts hourly files into a single file for each hour.
    """
    logger.info(f'\n********** Compacting day *************')
    logger.info(f'****Src uri****** {base_uri}') 
    logger.info(f'****Dst uri****** {compact_base_uri}') 
    bucket_and_prefix = base_uri.split('//')[1]
    bucket = bucket_and_prefix.split('/')[0]
    prefix = '/'.join(bucket_and_prefix.split('/')[1:])
    logger.info(f'****prefix****** {prefix}')  
    
    if partition_mode == partition_day:
        compact_files(spark_session, base_uri, compact_base_uri)
    else:
        try:
            folders = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
            if folders and folders.get('CommonPrefixes'):
                for obj in folders.get('CommonPrefixes'):
                    prefix = obj.get('Prefix')            
                    folder = prefix.split('=')
                    if 'hour' in folder[-2]:
                        hour = int(folder[-1].split('/')[0])
                        src_uri = f'{base_uri}hour={hour:02d}'
                        dst_uri = f'{compact_base_uri}hour={hour:02d}' 
                        compact_files(spark_session, src_uri, dst_uri)
            else:
                logger.info(f'Prefix does not have any data to compact')  
        except botocore.client.ClientError as error:
                logger.info(f'\n No bucket Exception caught: {error.response} for bucket {bucket}')  

def compact_full(spark_session: SparkSession, base_uri: str, compact_base_uri: str, partition_mode: str):
    """
    When the mode is full, this method will process all data in the feature group by recursivley traversing the prefixes. 
    """    
    logger.info(f'\n********** Compacting full *************')
    bucket_and_prefix = base_uri.split('//')[1]
    bucket = bucket_and_prefix.split('/')[0]
    prefix = '/'.join(bucket_and_prefix.split('/')[1:])
    full_traverse(spark_session, base_uri, compact_base_uri, bucket, prefix, partition_mode)
    
def compact_incremental(spark_session: SparkSession, base_uri: str, compact_base_uri: str, partition_mode: str):
    """
    When the mode is incremental, this method will process previous day's data. Assuming this is run at 12 AM at night as a 
    batch job or run the next day. 
    """
    logger.info(f'\n********** Compacting incremental *************')
    previous_date = datetime.today() - timedelta(days=1)
    src_uri = base_uri + f'year={previous_date.year}/month={previous_date.month:02d}/day={previous_date.day:02d}/'
    dst_uri = compact_base_uri + (f'year={previous_date.year}/month={previous_date.month:02d}'
                                f'/day={previous_date.day:02d}/')
    compact_day(spark_session, src_uri, dst_uri, partition_mode)
        
    
def compact_offline_store(spark_session: SparkSession,
                          base_uri: str, compact_base_uri: str, 
                          year: int, month:int, day:int, compact_mode: str, partition_mode: str) -> None:
    logger.info(f'\n  ***** Compact Mode *******={compact_mode}\n') 
    logger.info(f'\n  ***** Partition Mode *******={partition_mode}\n') 
    
    if not base_uri and not compact_base_uri:
        logger.info(f'\n  Input or Output S3 URI not specified \n') 
        return
    
    # Check if year, month, day has been provided as inputs
    if compact_mode == mode_day and year and month and day:
        logger.info(f'\n  Compacting specific day \n') 
        src_uri = base_uri + f'year={year}/month={month:02d}/day={day:02d}/'
        dst_uri = compact_base_uri + f'year={year}/month={month:02d}/day={day:02d}/'
        compact_day(spark_session, src_uri, dst_uri, partition_mode)
        
    # Check if mode is set to incremental
    elif compact_mode == mode_incremental:
        logger.info(f'\n  Compacting single day\n')  
        compact_incremental(spark_session, base_uri, compact_base_uri, partition_mode)
        
    # Check if mode is set to incremental
    elif compact_mode == mode_full:
        logger.info(f'\n  Compacting entire feature group \n')  
        compact_full(spark_session, base_uri, compact_base_uri, partition_mode)
    return

def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_group_name', type=str)
    parser.add_argument('--region_name', type=str)
    parser.add_argument('--s3_input_uri_prefix', type=str)
    parser.add_argument('--s3_output_uri_prefix', type=str)
    parser.add_argument('--year', type=int)
    parser.add_argument('--month', type=int)
    parser.add_argument('--day', type=int)
    parser.add_argument('--compact_mode', type=str,default='incremental')    
    parser.add_argument('--partition_mode', type=str,default='hour')    
    args, _ = parser.parse_known_args()
    return args

def run_spark_job():
    args = parse_args()
    
    spark_session = SparkSession.builder.appName('PySparkJob').getOrCreate()
    spark_context = spark_session.sparkContext
    
    total_cores = int(spark_context._conf.get('spark.executor.instances')) * \
                  int(spark_context._conf.get('spark.executor.cores'))
    logger.info(f'Total available cores in the Spark cluster = {total_cores}')
    
    compact_offline_store(spark_session,
                          args.s3_input_uri_prefix, 
                          args.s3_output_uri_prefix,
                          args.year,
                          args.month,
                          args.day,
                          args.compact_mode,
                          args.partition_mode)

if __name__ == '__main__':
    logger.info('COMPACTION STARTED')
    start_time = time.time()
    run_spark_job()
    logger.info('COMPACTION COMPLETED')
    logger.info(f'\n TOOK {int(time.time() - start_time):,d} SECONDS')