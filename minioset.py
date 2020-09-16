import os
import boto3
from zipfile import ZipFile


def compress_object(zippath, dirpath):
    print("compress {} on {}".format(zippath, dirpath))
    with ZipFile(zippath, "w") as zipfile:
        for rootpath, dirs, files in os.walk(dirpath):
            for filename in files:
                zipfile.write(os.path.join(rootpath, filename))
    return zippath


def uncompress_object(zippath, dirpath=os.getcwd()):
    print("uncompress {} on {}".format(zippath, dirpath))
    zipfile = ZipFile(zippath)
    for filename in zipfile.namelist():
        zipfile.extract(filename, dirpath)
    zipfile.close()


def save_object(client, bucketname, objectpath):
    print("save {} on minio({})".format(objectpath, bucketname))
    with open(objectpath, "rb") as objectfile:
        client.put_object(Bucket=bucketname, Key=objectpath, Body=objectfile)


def load_object(client, bucketname, objectpath):
    print("load {} on minio({})".format(objectpath, bucketname))
    s3object = client.get_object(Bucket=bucketname, Key=objectpath)
    with open(objectpath, "wb") as objectfile:
        objectfile.write(s3object["Body"].read())


def connect_server(endpointurl, accesskey, secretkey, regionname=None):
    client = boto3.client(
        "s3",
        endpoint_url=endpointurl,
        aws_access_key_id=accesskey,
        aws_secret_access_key=secretkey,
        region_name=regionname,
    )
    return client
